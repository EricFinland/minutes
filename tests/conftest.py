# onnxruntime must be imported before PyQt6 or it crashes on load (see app.py,
# _preload_native_libs). The UI tests import Qt, so load it first for the whole run.
import onnxruntime  # noqa: F401
