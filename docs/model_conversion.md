# Model Conversion Pipeline

This document describes the process of converting machine learning models from popular frameworks to DirectML format for use on Xbox Series S.

## Overview

Converting models from frameworks like PyTorch or TensorFlow to DirectML involves several steps:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Original   │     │     ONNX     │     │  Optimized   │     │   DirectML   │
│    Model     │────▶│    Format    │────▶│     ONNX     │────▶│    Format    │
│ PyTorch/TF   │     │ Intermediate │     │   Model      │     │  Xbox Ready  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Conversion Steps

### 1. Export to ONNX

ONNX (Open Neural Network Exchange) serves as a standardized intermediate format:

#### From PyTorch:
```python
import torch
import torchvision

# Load your PyTorch model
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Create a sample input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,               # Model being exported
    dummy_input,         # Sample input
    "model.onnx",        # Output file
    export_params=True,  # Export parameters
    opset_version=11,    # ONNX version
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
```

#### From TensorFlow:
```python
import tensorflow as tf

# Load your TensorFlow model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Convert to ONNX
import tf2onnx
import onnx

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "model.onnx"

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11, output_path=output_path)
```

### 2. Optimize ONNX Model

Optimization simplifies the model graph and improves performance:

```python
import onnx
from onnxoptimizer import optimize

# Load the ONNX model
onnx_model = onnx.load("model.onnx")

# Apply optimizations
optimized_model = optimize(
    onnx_model,
    passes=[
        "eliminate_identity",
        "eliminate_nop_transpose",
        "fuse_consecutive_transposes",
        "fuse_bn_into_conv"
    ]
)

# Save the optimized model
onnx.save(optimized_model, "optimized_model.onnx")
```

### 3. Convert to DirectML Format

The final step converts the optimized ONNX model to DirectML:

```csharp
// C# code for conversion to DirectML
public class OnnxToDirectMLConverter
{
    public async Task<bool> ConvertOnnxToDirectML(string onnxPath, string directMLOutputPath)
    {
        // Load the ONNX model
        var onnxModel = await OnnxModel.LoadFromFileAsync(onnxPath);
        
        // Map ONNX operations to DirectML
        var directMLModel = new DirectMLModel();
        
        foreach (var node in onnxModel.Nodes)
        {
            // Map operation type
            var dmlOp = MapOnnxToDml(node.OpType);
            
            // Configure operation attributes
            ConfigureOperationAttributes(dmlOp, node.Attributes);
            
            // Add to model
            directMLModel.AddOperation(dmlOp);
        }
        
        // Set up input and output bindings
        ConfigureInputOutputBindings(directMLModel, onnxModel);
        
        // Optimize for Xbox Series S
        OptimizeForXbox(directMLModel);
        
        // Save to file
        return await directMLModel.SaveToFileAsync(directMLOutputPath);
    }
    
    private DirectMLOperator MapOnnxToDml(string onnxOpType)
    {
        // Mapping logic from ONNX to DirectML operators
        switch (onnxOpType)
        {
            case "Conv": return new DirectMLConvOperator();
            case "MaxPool": return new DirectMLPoolingOperator(PoolingType.Max);
            case "Relu": return new DirectMLActivationOperator(ActivationType.Relu);
            // Additional mappings...
            default: throw new NotSupportedException($"Operation {onnxOpType} not supported");
        }
    }
    
    // Other implementation methods...
}
```

## Quantization

To improve performance and reduce memory usage, models can be quantized:

- **FP32 to FP16 conversion**: Reduces memory footprint by 50%
- **INT8 quantization**: Further reduces size at some accuracy cost
- **Post-training quantization**: Calibrates quantization based on representative data

```csharp
public void QuantizeToFP16(DirectMLModel model)
{
    foreach (var weight in model.Weights)
    {
        // Convert FP32 tensor to FP16
        weight.ConvertToFP16();
    }
    
    // Update model to use FP16 operations where possible
    model.OptimizeForPrecision(PrecisionLevel.FP16);
}
```

## Pruning and Compression

Additional techniques to reduce model size:

- **Weight pruning**: Removing weights close to zero
- **Channel pruning**: Removing less important channels
- **Knowledge distillation**: Training smaller models to mimic larger ones

## Xbox-Specific Optimizations

Final optimizations for the Xbox Series S:

- **Memory layout adjustments**: Optimize tensor storage for RDNA 2 architecture
- **Operation fusion**: Combine multiple operations where possible
- **Shader specialization**: Generate specialized compute shaders for common operations

## Validation

After conversion, models must be validated:

```csharp
public async Task<bool> ValidateModelConversion(string originalOnnxPath, string convertedDmlPath)
{
    // Load test data
    var testInputs = LoadValidationData();
    
    // Run inference on original ONNX model
    var onnxResults = await RunOnnxInference(originalOnnxPath, testInputs);
    
    // Run inference on converted DirectML model
    var dmlResults = await RunDirectMLInference(convertedDmlPath, testInputs);
    
    // Compare results
    return CompareResults(onnxResults, dmlResults, threshold: 1e-4f);
}
```

## Supported Model Types

The current conversion pipeline supports:

- Image classification (ResNet, MobileNet, EfficientNet)
- Object detection (SSD, YOLO-tiny)
- Image segmentation (lightweight U-Net variants)
- Natural language processing (small BERT models, DistilBERT)
- Audio processing (keyword spotting, limited speech recognition) 