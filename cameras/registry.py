"""
cameras/registry.py
===================
Maps a handler name in config/cameras.py to its handler class.

To add a new camera type later: create cameras/<name>.py with a HANDLER
class, then add one line here.
"""
import importlib

_HANDLERS = {
    "reception": "cameras.reception",     # entry counting
    "workspace": "cameras.workspace",     # presence / time-in-area
}


def build_camera(cfg):
    handler = cfg.get("handler")
    module_path = _HANDLERS.get(handler)
    if module_path is None:
        raise ValueError(
            f"Unknown handler '{handler}' for camera '{cfg.get('key')}'. "
            f"Known handlers: {list(_HANDLERS)}")
    module = importlib.import_module(module_path)
    return module.HANDLER(cfg)
