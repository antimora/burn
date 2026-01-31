# Relu

Since opset **14**

## Description

Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.

## Inputs (1 - 1)

- **X** (T): Input tensor

## Outputs (1 - 1)

- **Y** (T): Output tensor

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8)
  Constrain input and output types to signed numeric tensors.
