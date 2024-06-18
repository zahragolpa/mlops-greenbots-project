"""
Converts models to ONNX and saves them.

To be usable, the models need to be serialized
by using `save_onnx_model`
"""

import logging
from typing import Any

import joblib
from skl2onnx import to_onnx

# Add this import so it can be imported via this file
from skl2onnx.helpers.onnx_helper import save_onnx_model # noqa

logger = logging.getLogger(__name__)


def convert_joblib_sklearn_to_onnx(classifier: str|Any, X: Any, **kwargs):
    """Transforms a classifier into an ONNX model
    
    Note that in order to use the model, you will
    need to use the ONNX runtime instead.

    Returns:
        ONNX model: ONNX Representation of the model
    """
    if isinstance(classifier, str):
        logger.debug("Provided str type to load model, loading from path")
        classifier = joblib.load(classifier)

    # TODO(Participant): Use the proper method from `skl2onnx` to convert to an ONNX model
    onnx_model = to_onnx(classifier, X=X, **kwargs)

    return onnx_model

