"""
core/gpu.py
===========
Make onnxruntime's CUDA provider actually loadable on Windows.

The trap this exists for
------------------------
    >>> import onnxruntime as ort; print(ort.get_available_providers())
    ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

That list is what onnxruntime was BUILT with, not what it can load. Ask
for CUDAExecutionProvider anyway and you get:

    Error loading "...onnxruntime_providers_cuda.dll" which depends on
    "cublasLt64_12.dll" which is missing.

...printed to stderr, followed by a SILENT fall back to the CPU. Nothing
raises. So face recognition keeps working, perfectly correctly, at about
twenty times the cost - and on this project that means the face pass
cannot keep up, faces almost never reach the gallery in time, and
everybody ends up labelled "Unknown" for reasons that look like a
threshold problem rather than a missing DLL.

Why the DLLs are missing, and why nothing needs installing
---------------------------------------------------------
They are not missing. torch ships its own copy of the whole CUDA runtime
in site-packages/torch/lib - cublasLt64_12.dll and cudnn64_9.dll included,
which is exactly what onnxruntime-gpu wants. They simply are not on the
search path when onnxruntime loads its provider.

os.add_dll_directory() is NOT enough on its own: onnxruntime loads the
provider DLL by full path, and the provider's own dependencies are then
resolved by the standard Windows search order, which ignores directories
added that way. Putting the directory on PATH is what works, and it has
to happen BEFORE onnxruntime is imported.
"""
import os
import sys
import glob

_done = False


def _candidate_dirs():
    """Directories that may hold the CUDA runtime DLLs, best first."""
    site = os.path.join(sys.prefix, "Lib", "site-packages")
    dirs = [os.path.join(site, "torch", "lib")]
    # pip-installed nvidia-* wheels keep theirs here instead
    dirs += sorted(glob.glob(os.path.join(site, "nvidia", "*", "bin")))
    return [d for d in dirs if os.path.isdir(d)]


def enable_onnx_cuda(verbose=True):
    """Put the CUDA runtime on the DLL search path. Safe to call twice.

    Call this BEFORE importing onnxruntime or insightface. Returns the
    directories that were added.
    """
    global _done
    if _done or not sys.platform.startswith("win"):
        return []

    added = []
    for d in _candidate_dirs():
        if not glob.glob(os.path.join(d, "*.dll")):
            continue
        # PATH is the part that actually matters (see the note above);
        # add_dll_directory covers anything loaded the other way.
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
        added.append(d)

    _done = True
    if verbose and "onnxruntime" in sys.modules:
        print("[GPU] warning: onnxruntime was already imported before the "
              "CUDA runtime was put on the path - it may have fallen back "
              "to the CPU. Import core.gpu earlier.")
    return added


def provider_of(session_owner):
    """Which provider a loaded InsightFace model is REALLY using.

    Used so the console can report the truth instead of assuming GPU.
    """
    try:
        return session_owner.session.get_providers()[0]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------
# Shared ONNX Runtime session builder
#
# SCRFD and AdaFace are both plain ONNX graphs that we drive ourselves,
# so both want the same thing: the CUDA runtime on the path first, a
# quiet session (the provider fallback is extremely noisy on stderr and
# scrolls the real startup messages away), and an honest report of which
# provider actually loaded.
#
# TensorRT is offered but never forced. The TRT execution provider
# compiles the graph on first use, which can take minutes and is cached
# per GPU and per input shape; that is a fine trade for a fixed-shape
# recognition model on a machine that stays up for weeks, and a bad one
# on a laptop being debugged. TRT_CACHE_DIR keeps the compiled engines
# so only the first run pays.
# ---------------------------------------------------------------------

def onnx_providers(use_tensorrt=False, cache_dir=None, device_id=0):
    """Provider list, best first, in the form onnxruntime wants."""
    providers = []
    if use_tensorrt:
        trt = {"device_id": int(device_id),
               "trt_fp16_enable": True,
               "trt_engine_cache_enable": True}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            trt["trt_engine_cache_path"] = cache_dir
        providers.append(("TensorrtExecutionProvider", trt))
    providers.append(("CUDAExecutionProvider", {"device_id": int(device_id)}))
    providers.append("CPUExecutionProvider")
    return providers


def onnx_session(model_path, use_tensorrt=False, cache_dir=None, device_id=0,
                 intra_threads=0):
    """Build an InferenceSession. Returns (session, provider_name).

    Falls back through TensorRT -> CUDA -> CPU on its own; the returned
    provider name is what actually loaded, so the caller can say so out
    loud rather than assuming.
    """
    enable_onnx_cuda(verbose=False)
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3          # the fallback chatter is noise
    options.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_threads:
        options.intra_op_num_threads = int(intra_threads)

    # A LADDER, not one attempt.
    #
    # This matters more than it looks. When the TensorRT provider cannot
    # load its DLLs, onnxruntime does not raise and it does not quietly
    # skip to the next provider in the list - it abandons the whole list
    # and retries with the CPU alone. Asking for TensorRT on a machine
    # without it therefore costs you CUDA as well, and face recognition
    # ends up on the CPU while the console cheerfully reports success.
    # Stepping down one rung at a time is what stops an optional
    # accelerator taking the mandatory one with it.
    ladder = []
    if use_tensorrt:
        ladder.append(onnx_providers(True, cache_dir, device_id))
    ladder.append(onnx_providers(False, cache_dir, device_id))
    ladder.append(["CPUExecutionProvider"])

    session = None
    for providers in ladder:
        try:
            session = ort.InferenceSession(model_path, options,
                                           providers=providers)
        except Exception:
            continue
        used = session.get_providers()
        wanted_first = providers[0]
        wanted_name = (wanted_first[0] if isinstance(wanted_first, tuple)
                       else wanted_first)
        if used and (used[0] == wanted_name
                     or wanted_name == "CPUExecutionProvider"):
            return session, used[0]
        # It loaded, but not on the provider we asked for - that is the
        # silent-fallback case. Drop a rung and try again.
    if session is None:
        session = ort.InferenceSession(model_path, options,
                                       providers=["CPUExecutionProvider"])
    used = session.get_providers()
    return session, (used[0] if used else "unknown")
