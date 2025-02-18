import onnx
from onnx import shape_inference

# Load your ONNX model
# model = onnx.load('onnx-tests/tests/expand/expand_shape.onnx')
model = onnx.load('onnx-tests/tests/conv1d/conv1d.onnx')

# Apply shape inference
inferred_model = shape_inference.infer_shapes(model)

# Save the inferred model
onnx.save(inferred_model, 'conv1d_inferred.onnx')
