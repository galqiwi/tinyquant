from typing import Any, Dict

from tinyquant.quantized_linear import dequantize_meta, quantize_meta


def check_quantize_dequantize(meta: Dict[str, Any]) -> None:
    assert dequantize_meta(quantize_meta(meta)) == meta


def test_meta() -> None:
    check_quantize_dequantize(dict())
    check_quantize_dequantize({"foo": "bar"})
    check_quantize_dequantize({"foo": 1, "bar": None})
