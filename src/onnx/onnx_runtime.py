"""
Loads ONNX models into the appropriate runtime

Separate file than the onnx_save module because,
ONNX runtime is not tied to saving the model:
we could save the model from anywhere else and
loaded here.
"""

import onnxruntime as rt



def create_onnx_session(path_or_bytes: str|bytes) -> rt.InferenceSession:
    sess = rt.InferenceSession(path_or_bytes, providers=rt.get_available_providers())
    return sess
