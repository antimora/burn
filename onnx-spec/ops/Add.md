# Add

Since opset **14**

## Description

Performs element-wise binary addition (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

## Inputs (2 - 2)

- **A** (T): First operand.
- **B** (T): Second operand.

## Outputs (1 - 1)

- **C** (T): Result, has same element type as two inputs

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to all numeric tensors.
