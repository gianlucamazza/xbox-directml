# Setting Up a Pre-trained Model for DirectML

This guide explains how to obtain and prepare a pre-trained model for use with this application.

## Quick Option: Download Pre-converted ResNet-18

For a quick start, you can download a pre-converted ResNet-18 model in DirectML format:

1. Visit the [DirectML GitHub examples](https://github.com/microsoft/DirectML)
2. Download a pre-converted model if available
3. Rename it to `resnet18.dml` and place it in this folder

## Converting Your Own Model

### Step 1: Install Required Tools

First, ensure you have the necessary Python packages:

```bash
pip install torch torchvision onnx onnxruntime onnxoptimizer
```

### Step 2: Export a PyTorch Model to ONNX

```python
import torch
import torchvision.models as models

# Load a pre-trained ResNet-18
model = models.resnet18(pretrained=True)
model.eval()

# Create a sample input (RGB image, 224x224 pixels)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,                     # model being exported
    dummy_input,               # model input
    "resnet18.onnx",           # output file
    export_params=True,        # store the trained parameters
    opset_version=11,          # ONNX version to export
    do_constant_folding=True,  # optimize constants
    input_names=["input"],     # model's input names
    output_names=["output"],   # model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model exported to ONNX format!")
```

### Step 3: Optimize the ONNX Model

```python
import onnx
from onnxoptimizer import optimize

# Load the ONNX model
onnx_model = onnx.load("resnet18.onnx")

# Verify the model structure
onnx.checker.check_model(onnx_model)

# Apply optimizations
optimized_model = optimize(
    onnx_model,
    passes=[
        "eliminate_identity",
        "eliminate_nop_transpose",
        "fuse_consecutive_transposes",
        "fuse_bn_into_conv",
        "fuse_pad_into_conv",
        "fuse_add_bias_into_conv",
    ]
)

# Save the optimized model
onnx.save(optimized_model, "resnet18_optimized.onnx")

print("ONNX model optimized!")
```

### Step 4: Use with DirectML

There are two approaches:

#### Option A: Use ONNX Model Directly with DirectML

The current placeholder code is set up to use a custom DirectML format, but in a real implementation, you could load the ONNX model directly with code like:

```cpp
// Using ONNXRuntime with DirectML backend
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DirectMLInference");
Ort::SessionOptions session_options;

// Enable DirectML
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));

// Create session
Ort::Session session(env, "resnet18_optimized.onnx", session_options);
```

#### Option B: Convert to Custom DirectML Format

For more control, implement a custom model format:

1. Parse the ONNX model
2. Create DirectML operators for each ONNX node
3. Compile and save as a custom format

## ImageNet Labels

This project includes `imagenet_labels.txt` with common ImageNet class labels. If your model uses different labels, replace this file with the appropriate class names.

## Testing the Model

Once you have your model:

1. Ensure it's named `resnet18.dml` and placed in this folder
2. The app should load it automatically
3. Test with various images to validate correct operation

## Troubleshooting

- If the model fails to load, check the exact format requirements in the DirectML documentation
- ONNX models can be validated with tools like Netron (https://netron.app/)
- For complex models, you may need to simplify or quantize to fit within Xbox memory constraints 