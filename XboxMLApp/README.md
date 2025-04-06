# Xbox DirectML Machine Learning App

This project demonstrates how to use DirectML on Xbox Series S to perform machine learning inference.

## Project Structure

```
XboxMLApp/
├── Assets/                   # Images and assets
├── Models/                   # ML models and label files
│   ├── resnet18.dml          # Pre-converted ResNet-18 model
│   └── imagenet_labels.txt   # ImageNet class labels
├── Source/                   # Source code
│   ├── UI/                   # XAML UI components
│   ├── Core/                 # Core C++ implementation
│   │   ├── DirectML/         # DirectML implementation
│   │   ├── Models/           # Model management
│   │   └── Utils/            # Utilities
│   └── Interop/              # C#/C++ interoperability
```

## Building the Project

1. Open the solution in Visual Studio 2022
2. Ensure the target platform is set to x64
3. Build the solution (Ctrl+Shift+B)

## Deploying to Xbox

### Prerequisites
- Xbox Series S with Developer Mode activated
- Visual Studio 2022 with UWP and Xbox development support
- Network connection between PC and Xbox

### Configure Xbox Connection

1. In Visual Studio, go to Project > Properties
2. Under Debug tab:
   - Set "Target device" to "Remote Machine"
   - Enter your Xbox's IP address 
   - Authentication Mode: "Universal (Unencrypted Protocol)"

### Deploy and Run

1. In Visual Studio, set the build configuration to "Debug" and platform to "x64"
2. Select "Remote Machine" from the dropdown next to the Run button
3. Press F5 to build, deploy, and run the app on your Xbox

## Converting Your Own Models

This sample includes a placeholder ResNet-18 model. To use your own models:

1. Export your trained model from PyTorch or TensorFlow to ONNX format
2. Use the optimizations described in the docs to prepare for DirectML
3. Replace the models/resnet18.dml file with your converted model
4. Update the models/imagenet_labels.txt file if using a different classification scheme

## Known Issues

- This is a demonstration project and uses placeholder implementations
- For a production application, you would need complete model loading and DirectML implementation

## Additional Resources

- [DirectML GitHub](https://github.com/microsoft/DirectML)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [Xbox Developer Documentation](https://developer.microsoft.com/en-us/games/xbox/) 