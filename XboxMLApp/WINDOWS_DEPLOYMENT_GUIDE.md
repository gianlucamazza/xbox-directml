# Windows Deployment Guide for Xbox ML App

This guide walks you through deploying the DirectML application to your Xbox Series S from a Windows machine.

## Prerequisites

1. Windows 10/11 PC
2. Visual Studio 2022 with the following workloads installed:
   - Universal Windows Platform development
   - Game development with C++
   - .NET desktop development

3. Xbox Series S with Developer Mode activated
4. Network connection between your PC and Xbox

## Step 1: Set Up Your Windows Environment

1. Install Visual Studio 2022 (Community edition is sufficient)
2. During installation, select the workloads mentioned in prerequisites
3. Ensure you have the Windows 10/11 SDK (version 10.0.19041.0 or later)
4. Install the DirectML SDK via NuGet (will be handled by the project)

## Step 2: Prepare Your Xbox

1. Activate Developer Mode on your Xbox Series S:
   - Install the Dev Mode Activation app from the Xbox Store
   - Launch the app and follow the on-screen instructions

2. Configure your Xbox for development:
   - After rebooting into Developer Mode, note the Xbox's IP address (shown on screen)
   - In the Developer Home, go to Settings:
     - Enable Xbox Device Portal
     - Set memory allocation to at least 6GB for apps
     - Configure network settings if needed

## Step 3: Unpack and Open the Project

1. Extract `XboxMLApp-Deployment.tar.gz` to a folder on your Windows PC
2. Open Visual Studio 2022
3. Open the solution file: `XboxMLApp\XboxMLApp.sln`
4. Wait for NuGet packages to restore

## Step 4: Configure the Project for Xbox Deployment

1. Right-click on the XboxMLApp project in Solution Explorer > Properties
2. Under Debug tab:
   - Set "Target device" to "Remote Machine"
   - Enter your Xbox's IP address in "Remote machine name"
   - Authentication Mode: "Universal (Unencrypted Protocol)"
   - In "Deployment Options", ensure "Remove previous deployment" is checked

## Step 5: Build and Deploy

1. Set the Solution Configuration to "Debug"
2. Set the Solution Platform to "x64"
3. Press F5 or select "Debug > Start Debugging"
4. Visual Studio will:
   - Build the application
   - Package it
   - Deploy it to your Xbox
   - Launch it

## Step 6: Using the Application

1. The app will run on your Xbox
2. Use the Xbox controller to navigate:
   - Press A to select buttons
   - Use the D-pad or left stick to navigate UI elements
3. Select "Load Image" to choose an image
4. Select "Run Inference" to process the image with DirectML

## Troubleshooting

### If deployment fails:
- Verify your Xbox is in Developer Mode
- Check the IP address is correct
- Ensure the Xbox Device Portal is enabled
- Check your network connection
- Try restarting both your Xbox and Visual Studio

### If the app builds but crashes:
- Check Output and Error List windows in Visual Studio for details
- Consider enabling native code debugging (Project Properties > Debug > Enable native code debugging)
- Look at Debug output for DirectML-specific errors

### If the model doesn't load:
- Ensure the model file is correctly placed and built as content
- Check the app's temporary storage for log files
- Try using a simpler test model first

## Advanced: Converting Your Own Models

To use your own ML models:

1. Export from PyTorch/TensorFlow to ONNX format:
   ```python
   # PyTorch example
   import torch
   model = torch.load('your_model.pth')
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy_input, 'model.onnx')
   ```

2. Optimize the ONNX model (optional but recommended):
   ```python
   import onnx
   from onnxoptimizer import optimize
   
   model = onnx.load('model.onnx')
   optimized = optimize(model)
   onnx.save(optimized, 'optimized_model.onnx')
   ```

3. Convert to DirectML format or use ONNX directly with DirectML
4. Replace Models/resnet18.dml with your converted model
5. Update class labels if needed

## Resources

- [Xbox UWP Development Documentation](https://docs.microsoft.com/en-us/windows/uwp/xbox-apps/)
- [DirectML GitHub](https://github.com/microsoft/DirectML)
- [ONNX Model Zoo](https://github.com/onnx/models) (pre-trained models)
- [PIX for Windows](https://devblogs.microsoft.com/pix/) (Graphics debugging) 