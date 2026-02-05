# How to implement new quantization method for tinyquant

This documentation describes how to implement a new quantization method for tinyquant library.

To describe a new quantization method, you need to define two things
- How to do matmul on quantized matrix representation
- How to quantize a matrix

To be precise, the former is needed for inference, and latter is needed for quantization process.
You are free to only implement matmul, if you prefer out-of-framework quantization method.

This is how this is represented in tinyquant. There are only two classes needed to implement an inference:
- `QuantizedLinear` is an `nn.Module` that contains just one `nn.ParameterDict`.
It contains all tensors that are needed for quantized representation.
- `Quantizer` is an interface that defines matmul operation `forward: (QuantizedLinear, Tensor) -> Tensor`.
It needs to be implemented for every quantization method.

Currently, only data free quantizations can be implemented in-framework.
To do that, one should use `DataFreeQuantizer` class instead of `Quantized`.
This class defines data free quantization method that returns `QuantizedLinear` from weight `Tensor`.

You can learn more by looking into nf4 quantization. Good starting point is
```
grep -r tinyquant -e nf4
```