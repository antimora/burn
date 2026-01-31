# Clip

Since opset **13**

## Description

Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
When 'min' is greater than 'max', the clip operator sets all the 'input' values to
the value of 'max'. Thus, this is equivalent to 'Min(max, Max(input, min))'.

## Inputs (1 - 3)

- **input** (T): Input tensor whose elements to be clipped
- **min** (T, optional): Minimum value, under which element is replaced by min. It must be a scalar(tensor of empty shape).
- **max** (T, optional): Maximum value, above which element is replaced by max. It must be a scalar(tensor of empty shape).

## Outputs (1 - 1)

- **output** (T): Output tensor with clipped input elements

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to all numeric tensors.
