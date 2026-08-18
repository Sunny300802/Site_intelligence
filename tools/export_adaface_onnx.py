"""
tools/export_adaface_onnx.py
============================
Convert a released AdaFace checkpoint into the ONNX file the pipeline
runs. You do this ONCE per model.

    python tools/export_adaface_onnx.py --ckpt adaface_ir101_webface12m.ckpt

Writes data/models/adaface_ir101_webface12m.onnx, which core/face_embed.py
finds automatically.

WHERE TO GET A CHECKPOINT
-------------------------
AdaFace publishes several. For a CCTV site the useful ones are:

    adaface_ir101_webface12m.ckpt   best accuracy; trained on the largest
                                    and most varied set, which is what
                                    matters for poor-quality faces
    adaface_ir101_ms1mv3.ckpt       very close, different training set
    adaface_ir50_ms1mv2.ckpt        about half the cost, a little weaker

Download from the AdaFace project's own release page. This tool does not
fetch anything - a face recognition model should be something you chose
and can point at, not something a script pulled from wherever.

WHY THE OUTPUT IS CHECKED
-------------------------
The exported graph is run against the PyTorch model on a fixed input and
the two are compared. A conversion that is subtly wrong - the wrong
architecture, a mis-set dropout, a channel order slipped somewhere -
does not crash. It produces embeddings that are internally consistent
and match nobody, which looks exactly like a threshold problem and can
cost a week. The check is cheap and it makes that outcome impossible.

AFTERWARDS
----------
Re-enroll, because embeddings from two different models are not
comparable:

    python tools/enroll_faces.py
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch's newer exporter prints progress with emoji. On a Windows
# console that is cp1252 and the export dies on a UnicodeEncodeError
# after all the real work is done, which is a maddening way to lose a
# five-minute conversion.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from config.settings import MODEL_DIR      # noqa: E402


def load_state_dict(ckpt_path):
    """The backbone weights out of a Lightning checkpoint.

    AdaFace trains with PyTorch Lightning, so every key is prefixed
    'model.' and the file also contains the classification head, the
    optimiser state and the training hyper-parameters. Only the backbone
    is wanted.
    """
    import torch
    checkpoint = torch.load(ckpt_path, map_location="cpu",
                            weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    weights = {key[len("model."):]: value for key, value in state.items()
               if key.startswith("model.")}
    return weights or state


def build_model(architecture, weights, repo=None):
    """The AdaFace backbone with the checkpoint loaded, in eval mode."""
    import torch

    if repo:
        # Prefer the official definition when the user points at a
        # checkout - it is authoritative, and a future architecture will
        # work here before this file knows about it.
        sys.path.insert(0, repo)
        import net                                   # noqa: F401
        model = net.build_model(architecture)
        print(f"[EXPORT] architecture from {repo}/net.py")
    else:
        from tools import adaface_net
        model = adaface_net.build(architecture)
        print(f"[EXPORT] architecture reconstructed locally "
              f"(tools/adaface_net.py)")

    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        # strict=False first so the mismatch can be REPORTED usefully
        # rather than raised as one opaque line, then refuse anyway.
        print(f"[EXPORT] *** the checkpoint does not fit this architecture ***")
        if missing:
            print(f"[EXPORT]   {len(missing)} missing key(s), "
                  f"e.g. {missing[:3]}")
        if unexpected:
            print(f"[EXPORT]   {len(unexpected)} unexpected key(s), "
                  f"e.g. {unexpected[:3]}")
        print(f"[EXPORT] try a different --arch, or pass --repo pointing at "
              f"an AdaFace checkout to use its own net.py.")
        raise SystemExit(2)

    model.eval()
    return model


def verify(model, onnx_path, tolerance=1e-3):
    """Does the exported graph agree with the PyTorch model?"""
    import numpy as np
    import torch
    from core.gpu import onnx_session

    sample = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        reference = model(sample)
    if isinstance(reference, (tuple, list)):
        reference = reference[0]
    reference = reference.cpu().numpy()

    session, provider = onnx_session(onnx_path, use_tensorrt=False)
    name = session.get_inputs()[0].name
    produced = session.run(None, {name: sample.numpy()})[0]

    difference = float(np.abs(reference - produced).max())
    similarity = float(np.mean(np.sum(
        reference / np.linalg.norm(reference, axis=1, keepdims=True) *
        produced / np.linalg.norm(produced, axis=1, keepdims=True), axis=1)))
    print(f"[EXPORT] verification on {provider}: "
          f"max difference {difference:.2e}, cosine {similarity:.6f}")
    return difference <= tolerance and similarity >= 0.999


def main():
    parser = argparse.ArgumentParser(
        description="Convert an AdaFace checkpoint to ONNX")
    parser.add_argument("--ckpt", required=True, help="the .ckpt file")
    parser.add_argument("--out", default=None,
                        help="output .onnx (default: data/models/<name>.onnx)")
    parser.add_argument("--arch", default=None,
                        help="ir_18 / ir_50 / ir_101 "
                             "(default: read from the filename)")
    parser.add_argument("--repo", default=None,
                        help="path to an AdaFace checkout, to use its own "
                             "net.py instead of the local reconstruction")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic-batch", action="store_true", default=True,
                        help="allow batched inference (default on - the "
                             "pipeline batches faces)")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[EXPORT] checkpoint not found: {args.ckpt}")
        return 1

    try:
        import torch
    except ImportError:
        print("[EXPORT] this tool needs PyTorch (the pipeline does not).")
        print("[EXPORT]   pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121")
        return 1

    from tools import adaface_net

    architecture = args.arch or adaface_net.guess_architecture(args.ckpt)
    if not architecture:
        print(f"[EXPORT] could not tell the architecture from "
              f"'{os.path.basename(args.ckpt)}'. Pass --arch ir_101 "
              f"(or ir_50 / ir_18).")
        return 1
    print(f"[EXPORT] {os.path.basename(args.ckpt)} as {architecture}")

    weights = load_state_dict(args.ckpt)
    print(f"[EXPORT] {len(weights)} backbone tensor(s) in the checkpoint")
    model = build_model(architecture, weights, args.repo)

    out_path = args.out
    if not out_path:
        stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        if not stem.startswith("adaface"):
            stem = f"adaface_{stem}"
        out_path = os.path.join(MODEL_DIR, f"{stem}.onnx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    wrapper = adaface_net.FeatureOnly(model)
    wrapper.eval()
    dummy = torch.randn(1, 3, 112, 112)
    dynamic = ({"input": {0: "batch"}, "embedding": {0: "batch"}}
               if args.dynamic_batch else None)

    print(f"[EXPORT] writing {out_path}")
    common = dict(input_names=["input"], output_names=["embedding"],
                  opset_version=args.opset, dynamic_axes=dynamic,
                  do_constant_folding=True)
    try:
        # The classic tracing exporter. This network is a plain feed
        # forward stack with no control flow, so tracing captures it
        # exactly, and it produces a smaller, more portable graph than
        # the dynamo path on the torch versions in the field.
        torch.onnx.export(wrapper, dummy, out_path, dynamo=False, **common)
    except TypeError:
        torch.onnx.export(wrapper, dummy, out_path, **common)   # older torch

    if verify(wrapper, out_path):
        print(f"[EXPORT] done - the exported model matches PyTorch.")
    else:
        # Leave the file in place but be unambiguous about it: a
        # near-miss is usually an opset issue, and the operator needs to
        # know NOT to enroll against this.
        print(f"[EXPORT] *** the exported model does NOT match PyTorch. ***")
        print(f"[EXPORT] do not enroll against this file. Try a different "
              f"--opset (14 or 17), or export from the official repo.")
        return 2

    print(f"\n[EXPORT] next steps:")
    print(f"[EXPORT]   1. python tools/enroll_faces.py "
          f"(embeddings from a different model are not comparable)")
    print(f"[EXPORT]   2. python tools/eval_recognition.py "
          f"--dataset data/faces --compare")
    print(f"[EXPORT]      to measure the right FACE_RECOGNITION_THRESHOLD "
          f"for your own people")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
