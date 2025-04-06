# Getting Started Guide

This guide helps you set up your development environment and run your first DirectML machine learning model on Xbox Series S.

## Prerequisites

Before you begin, ensure you have:

- Xbox Developer Account ($19)
- Xbox Series S with Developer Mode activated
- Windows 10/11 PC with Visual Studio 2022
- Stable network connection between PC and Xbox

## Development Environment Setup

### Step 1: Install Required Software

1. **Visual Studio 2022**
   - Community Edition is sufficient
   - Make sure to include these workloads during installation:
     - Universal Windows Platform development
     - Game development with C++
     - .NET desktop development

2. **Windows SDK**
   - Install Windows 10/11 SDK (version 10.0.19041.0 or later)
   - Include DirectX development tools

3. **DirectML SDK**
   - Download from NuGet or the Microsoft DirectML GitHub repository
   - Current recommended version: 1.13.0

### Step 2: Set Up Xbox Series S for Development

1. **Register as Xbox Developer**
   - Sign up at [Xbox Dev Mode Portal](https://dev.microsoft.com/xboxdevmode)
   - Pay the one-time $19 registration fee
   - Link your Xbox Series S console to your developer account

2. **Activate Developer Mode on Xbox**
   - Install Xbox Dev Mode Activation app from the Xbox Store
   - Follow the prompts to switch your console to Developer Mode
   - Make note of your console's IP address after reboot

3. **Configure Developer Settings**
   - In the Developer Home on Xbox, go to Settings
   - Set "Xbox Device Portal" to enabled
   - Configure Developer Network settings for remote deployment
   - Allow at least 6GB of memory for applications

### Step 3: Configure Visual Studio

1. **Create a UWP Project**
   - Open Visual Studio
   - Create a new project using the "Blank App (Universal Windows)" template
   - Set Target and Minimum Version to match your Windows SDK

2. **Configure Project Properties**
   - Right-click on the project in Solution Explorer > Properties
   - Under Debugging:
     - Set "Deployment Target" to "Remote Machine"
     - Enter your Xbox's IP address
     - Set "Authentication Mode" to "Universal (Unencrypted Protocol)"

3. **Add DirectML References**
   - Right-click on project References > Manage NuGet Packages
   - Search for and install the following packages:
     ```
     Microsoft.AI.DirectML (v1.13.0)
     Microsoft.Windows.CppWinRT (v2.0.220531.1)
     ```

## Your First DirectML Application

### Step 1: Project Structure Setup

Create the following folder structure in your project:

```
MyXboxMLApp/
├── Assets/               # For images, models, etc.
├── Content/              # For models and test data
├── Properties/
├── Source/
    ├── App.xaml         # UWP app entry point
    ├── MainPage.xaml    # UI definition
    ├── Core/            # C++ implementation
        ├── DirectML/    # DirectML integration
        ├── Models/      # Model handling code
        ├── Utils/       # Utility functions
```

### Step 2: Create the UI

Edit `MainPage.xaml` to create a simple UI:

```xml
<Page
    x:Class="MyXboxMLApp.MainPage"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    Background="{ThemeResource ApplicationPageBackgroundThemeBrush}">

    <Grid>
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>
        
        <TextBlock Grid.Row="0" Text="Xbox Series S DirectML Demo"
                   Style="{StaticResource HeaderTextBlockStyle}"
                   Margin="20"/>
        
        <Grid Grid.Row="1" Margin="20">
            <Grid.ColumnDefinitions>
                <ColumnDefinition Width="*"/>
                <ColumnDefinition Width="*"/>
            </Grid.ColumnDefinitions>
            
            <Border Grid.Column="0" BorderBrush="Gray" BorderThickness="2" Margin="10">
                <Grid>
                    <Grid.RowDefinitions>
                        <RowDefinition Height="Auto"/>
                        <RowDefinition Height="*"/>
                    </Grid.RowDefinitions>
                    <TextBlock Text="Input Image" Style="{StaticResource SubtitleTextBlockStyle}" Margin="10"/>
                    <Image x:Name="InputImage" Grid.Row="1" Stretch="Uniform" Margin="10"/>
                </Grid>
            </Border>
            
            <Border Grid.Column="1" BorderBrush="Gray" BorderThickness="2" Margin="10">
                <Grid>
                    <Grid.RowDefinitions>
                        <RowDefinition Height="Auto"/>
                        <RowDefinition Height="*"/>
                    </Grid.RowDefinitions>
                    <TextBlock Text="Results" Style="{StaticResource SubtitleTextBlockStyle}" Margin="10"/>
                    <ListView x:Name="ResultsList" Grid.Row="1" Margin="10"/>
                </Grid>
            </Border>
        </Grid>
        
        <StackPanel Grid.Row="2" Orientation="Horizontal" HorizontalAlignment="Center" Margin="20">
            <Button x:Name="LoadImageButton" Content="Load Image" Margin="10" Click="LoadImageButton_Click"/>
            <Button x:Name="RunModelButton" Content="Run Inference" Margin="10" Click="RunModelButton_Click"/>
        </StackPanel>
    </Grid>
</Page>
```

### Step 3: Create C++ Native Code for DirectML

Create a C++ header file `DirectMLManager.h` in the Core/DirectML folder:

```cpp
// DirectMLManager.h
#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <wrl/client.h>
#include <vector>
#include <string>

using Microsoft::WRL::ComPtr;

// Forward declarations
class Tensor;
struct TensorDesc;

class DirectMLManager
{
public:
    DirectMLManager();
    ~DirectMLManager();

    // Initialization and cleanup
    HRESULT Initialize();
    void Cleanup();
    
    // Model management
    HRESULT LoadModelFromFile(const std::wstring& modelPath);
    
    // Inference
    HRESULT RunInference(const float* inputData, size_t inputSize, 
                         float* outputData, size_t outputSize);

private:
    // DirectX resources
    ComPtr<ID3D12Device> m_d3d12Device;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    
    // DirectML resources
    ComPtr<IDMLDevice> m_dmlDevice;
    ComPtr<IDMLOperatorInitializer> m_dmlOperatorInitializer;
    ComPtr<IDMLBindingTable> m_dmlBindingTable;
    ComPtr<IDMLCompiledOperator> m_dmlModel;
    
    // Synchronization objects
    ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValue;
    HANDLE m_fenceEvent;
    
    // GPU resources
    ComPtr<ID3D12Resource> m_inputResource;
    ComPtr<ID3D12Resource> m_outputResource;
    
    // Helper methods
    HRESULT CreateDeviceResources();
    HRESULT ExecuteCommandList();
    HRESULT WaitForGpu();
};
```

### Step 4: Create C# Interop Layer

Create a C# wrapper class to interact with the C++ code:

```csharp
// DirectMLInterop.cs
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace MyXboxMLApp
{
    public class DirectMLInterop
    {
        // Native method imports
        [DllImport("DirectMLNative.dll")]
        private static extern IntPtr CreateDirectMLManager();
        
        [DllImport("DirectMLNative.dll")]
        private static extern void DestroyDirectMLManager(IntPtr manager);
        
        [DllImport("DirectMLNative.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool LoadModel(IntPtr manager, [MarshalAs(UnmanagedType.LPWStr)] string modelPath);
        
        [DllImport("DirectMLNative.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool RunInference(
            IntPtr manager,
            [In] float[] inputData, int inputSize,
            [Out] float[] outputData, int outputSize);
        
        // Member variables
        private IntPtr _managerHandle;
        
        // Constructor
        public DirectMLInterop()
        {
            _managerHandle = CreateDirectMLManager();
            if (_managerHandle == IntPtr.Zero)
            {
                throw new Exception("Failed to create DirectML Manager");
            }
        }
        
        // Destructor
        ~DirectMLInterop()
        {
            if (_managerHandle != IntPtr.Zero)
            {
                DestroyDirectMLManager(_managerHandle);
                _managerHandle = IntPtr.Zero;
            }
        }
        
        // Public methods
        public bool LoadModelFromFile(string modelPath)
        {
            return LoadModel(_managerHandle, modelPath);
        }
        
        public float[] RunInference(float[] inputData)
        {
            // For simplicity, we'll assume a fixed output size
            // In a real app, you'd query the model for its output shape
            float[] outputData = new float[1000]; // Assuming a classification model with 1000 classes
            
            bool success = RunInference(_managerHandle, inputData, inputData.Length, outputData, outputData.Length);
            return success ? outputData : null;
        }
    }
}
```

### Step 5: Download Test Model

For this example, we'll use a pre-trained ResNet-18 model:

1. Download a pre-converted DirectML ResNet-18 model from your internal repository or convert from ONNX using the conversion tools described in the model conversion documentation.

2. Place the model file in the `Content/Models` folder of your project.

3. Set its build action to "Content" and "Copy if newer" to ensure it's included in the deployment package.

### Step 6: Implement MainPage Logic

Add the implementation code to `MainPage.xaml.cs`:

```csharp
// MainPage.xaml.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media.Imaging;

namespace MyXboxMLApp
{
    public sealed partial class MainPage : Page
    {
        private DirectMLInterop _directML;
        private StorageFile _selectedImageFile;
        
        public MainPage()
        {
            this.InitializeComponent();
            
            // Initialize DirectML
            InitializeDirectMLAsync();
        }
        
        private async void InitializeDirectMLAsync()
        {
            try
            {
                _directML = new DirectMLInterop();
                
                // Load the model
                StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(
                    new Uri("ms-appx:///Content/Models/resnet18.dml"));
                
                bool modelLoaded = _directML.LoadModelFromFile(modelFile.Path);
                if (!modelLoaded)
                {
                    await ShowErrorMessageAsync("Failed to load the model");
                }
            }
            catch (Exception ex)
            {
                await ShowErrorMessageAsync($"Error initializing DirectML: {ex.Message}");
            }
        }
        
        private async void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FileOpenPicker();
            picker.ViewMode = PickerViewMode.Thumbnail;
            picker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
            picker.FileTypeFilter.Add(".jpg");
            picker.FileTypeFilter.Add(".jpeg");
            picker.FileTypeFilter.Add(".png");
            
            _selectedImageFile = await picker.PickSingleFileAsync();
            
            if (_selectedImageFile != null)
            {
                using (var stream = await _selectedImageFile.OpenAsync(FileAccessMode.Read))
                {
                    var bitmap = new BitmapImage();
                    await bitmap.SetSourceAsync(stream);
                    InputImage.Source = bitmap;
                }
            }
        }
        
        private async void RunModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedImageFile == null)
            {
                await ShowErrorMessageAsync("Please select an image first");
                return;
            }
            
            try
            {
                // Preprocess the image
                float[] inputTensor = await PreprocessImageAsync(_selectedImageFile);
                
                // Run inference
                float[] outputs = _directML.RunInference(inputTensor);
                
                if (outputs != null)
                {
                    // Process results (assuming classification model)
                    var classLabels = await LoadClassLabelsAsync();
                    var topResults = GetTopNResults(outputs, classLabels, 5);
                    
                    // Display results
                    ResultsList.Items.Clear();
                    foreach (var result in topResults)
                    {
                        ResultsList.Items.Add($"{result.Label}: {result.Confidence:P2}");
                    }
                }
            }
            catch (Exception ex)
            {
                await ShowErrorMessageAsync($"Error running inference: {ex.Message}");
            }
        }
        
        private async Task<float[]> PreprocessImageAsync(StorageFile imageFile)
        {
            // Load and resize image to 224x224 (standard for many models)
            // Convert to RGB float values normalized to [0,1]
            // This is simplified - actual preprocessing depends on the model
            
            // ...preprocessing code here...
            
            // For simplicity, return a dummy tensor of the right size
            return new float[3 * 224 * 224]; // RGB image data
        }
        
        private async Task<string[]> LoadClassLabelsAsync()
        {
            // Load ImageNet class labels
            StorageFile labelsFile = await StorageFile.GetFileFromApplicationUriAsync(
                new Uri("ms-appx:///Content/Models/imagenet_labels.txt"));
            
            var lines = await FileIO.ReadLinesAsync(labelsFile);
            return lines.ToArray();
        }
        
        private List<(string Label, float Confidence)> GetTopNResults(float[] outputs, string[] labels, int n)
        {
            // Find indices of top N values
            return outputs
                .Select((value, index) => (Label: labels[index], Confidence: value))
                .OrderByDescending(x => x.Confidence)
                .Take(n)
                .ToList();
        }
        
        private async Task ShowErrorMessageAsync(string message)
        {
            ContentDialog dialog = new ContentDialog
            {
                Title = "Error",
                Content = message,
                CloseButtonText = "OK"
            };
            
            await dialog.ShowAsync();
        }
    }
}
```

### Step 7: Deploy and Run

1. **Build Solution**
   - Make sure the build configuration is set to x64
   - Build the solution (Ctrl+Shift+B)

2. **Deploy to Xbox**
   - Select "Remote Machine" from the debug target dropdown
   - Press F5 to deploy and run

3. **Test Application**
   - Use the controller to navigate the UI
   - Load an image and run inference
   - View the results on screen

## Troubleshooting

### Common Issues and Solutions

1. **Deployment Fails**
   - Check that your Xbox is in Developer Mode
   - Verify the IP address in project properties
   - Ensure Xbox Device Portal is enabled on the console

2. **DirectML Errors**
   - Check that the model file is correctly included and deployed
   - Verify DirectML NuGet package is correctly referenced
   - Enable debug mode in DirectML for detailed error messages

3. **Performance Issues**
   - Try reducing model size through quantization
   - Check for memory leaks in native code
   - Use PIX for Windows to profile GPU usage

### Getting Help

- Visit the [DirectML GitHub repository](https://github.com/microsoft/DirectML) for issues and samples
- Check the [Xbox Developer Forums](https://forums.xboxlive.com/index.html) for platform-specific questions
- Search for error codes in the Microsoft DirectX documentation

## Next Steps

- Try running different types of models
- Experiment with the model conversion pipeline
- Implement more sophisticated preprocessing and UI
- Explore performance optimization techniques

For detailed information on these topics, refer to the other documentation sections. 