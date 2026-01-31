# Min

Since opset **13**

## Description

Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

## Inputs (1 - 2147483647)

- **data_0** (T, variadic): List of tensors for min.

## Outputs (1 - 1)

- **min** (T): Output tensor.

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to numeric tensors.
