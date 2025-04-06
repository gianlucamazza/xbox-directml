# DirectML Implementation

This document details the implementation of DirectML for machine learning on Xbox Series S.

## DirectML Overview

DirectML is Microsoft's hardware-accelerated machine learning API designed for DirectX 12. It provides low-level access to GPU compute capabilities while abstracting hardware-specific details.

Key DirectML features:
- Tight integration with DirectX 12
- Hardware-accelerated operators for common ML operations
- Optimized for Xbox and Windows platforms
- Support for both training and inference

## Integration Architecture

The DirectML integration follows this layered architecture:

```
┌───────────────────────────────────────────────────┐
│                UWP Application                     │
│                                                   │
│  ┌─────────────────┐        ┌──────────────────┐  │
│  │    C# UI Layer  │◄──────▶│  DirectML Interop │  │
│  └─────────────────┘        └──────────────────┘  │
│            △                         △            │
└────────────┼─────────────────────────┼────────────┘
             ▼                         ▼
┌─────────────────┐        ┌──────────────────────┐
│  C++ Core Logic │◄──────▶│ DirectML Manager C++ │
└─────────────────┘        └──────────────────────┘
                                     △
                                     │
                                     ▼
                           ┌──────────────────────┐
                           │   DirectX 12 API     │
                           └──────────────────────┘
                                     △
                                     │
                                     ▼
                           ┌──────────────────────┐
                           │  Xbox GPU Hardware   │
                           └──────────────────────┘
```

## Core Components

### DirectML Manager

The `DirectMLManager` class serves as the central coordinator:

```cpp
// DirectMLManager.h
class DirectMLManager
{
public:
    DirectMLManager();
    ~DirectMLManager();

    // Initialization
    HRESULT Initialize();
    
    // Model management
    HRESULT LoadModel(const std::wstring& modelPath);
    HRESULT UnloadModel();
    
    // Inference
    HRESULT ExecuteModel(const Tensor& inputTensor, Tensor& outputTensor);
    
    // Resource management
    HRESULT CreateTensor(const TensorDesc& desc, Tensor& tensor);
    HRESULT ReleaseTensor(Tensor& tensor);
    
private:
    // DirectX 12 resources
    ComPtr<ID3D12Device> m_d3d12Device;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    
    // DirectML resources
    ComPtr<IDMLDevice> m_dmlDevice;
    ComPtr<IDMLCommandRecorder> m_dmlCommandRecorder;
    ComPtr<IDMLOperatorInitializer> m_dmlOperatorInitializer;
    
    // Model-specific resources
    ComPtr<IDMLCompiledOperator> m_dmlCompiledModel;
    ComPtr<IDMLBindingTable> m_dmlBindingTable;
    
    // Helper methods
    HRESULT CreateDirectXResources();
    HRESULT CreateDirectMLResources();
    HRESULT CompileModel(const ModelDesc& desc);
    HRESULT BindTensors(const Tensor& inputTensor, Tensor& outputTensor);
    HRESULT ExecuteCommandList();
    HRESULT WaitForGPU();
};
```

### Tensor Implementation

The `Tensor` class abstracts tensor operations:

```cpp
// Tensor.h
class Tensor
{
public:
    Tensor();
    Tensor(const TensorDesc& desc);
    ~Tensor();
    
    // Data access
    HRESULT SetData(const void* data, size_t dataSize);
    HRESULT GetData(void* buffer, size_t bufferSize);
    
    // Tensor info
    const TensorDesc& GetDesc() const;
    uint64_t GetSizeInBytes() const;
    
    // GPU resources
    ID3D12Resource* GetResource() const;
    
private:
    TensorDesc m_desc;
    ComPtr<ID3D12Resource> m_resource;
    bool m_isGPUOwned;
    
    HRESULT CreateResource(ID3D12Device* device);
};

// TensorDesc.h
struct TensorDesc
{
    DML_TENSOR_DATA_TYPE dataType;
    std::vector<uint32_t> sizes;
    std::vector<uint32_t> strides;
    
    // Helper methods
    uint64_t GetSizeInBytes() const;
    DML_TENSOR_DESC GetDMLDesc() const;
};
```

## DirectML Initialization

Initializing DirectML involves setting up both DirectX 12 and DirectML resources:

```cpp
HRESULT DirectMLManager::Initialize()
{
    // Create DirectX 12 device
    HRESULT hr = CreateD3D12Device(
        nullptr,                   // Use default adapter
        D3D_FEATURE_LEVEL_11_0,    // Minimum feature level
        IID_PPV_ARGS(&m_d3d12Device)
    );
    if (FAILED(hr)) return hr;
    
    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    
    hr = m_d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue));
    if (FAILED(hr)) return hr;
    
    // Create command allocator and list
    hr = m_d3d12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&m_commandAllocator)
    );
    if (FAILED(hr)) return hr;
    
    hr = m_d3d12Device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_commandAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(&m_commandList)
    );
    if (FAILED(hr)) return hr;
    
    // Close the command list to prepare for first use
    hr = m_commandList->Close();
    if (FAILED(hr)) return hr;
    
    // Create DirectML device
    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
    
    hr = DMLCreateDevice(
        m_d3d12Device.Get(),
        dmlCreateDeviceFlags,
        IID_PPV_ARGS(&m_dmlDevice)
    );
    if (FAILED(hr)) return hr;
    
    // Create DirectML command recorder
    hr = m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlCommandRecorder));
    if (FAILED(hr)) return hr;
    
    // Create synchronization primitives
    // ...
    
    return S_OK;
}
```

## Model Loading Process

Loading a model involves these steps:

```cpp
HRESULT DirectMLManager::LoadModel(const std::wstring& modelPath)
{
    // Read model file
    std::vector<uint8_t> modelData;
    HRESULT hr = ReadFile(modelPath, modelData);
    if (FAILED(hr)) return hr;
    
    // Parse model header
    ModelHeader* header = reinterpret_cast<ModelHeader*>(modelData.data());
    
    // Extract weights
    std::vector<uint8_t> weights(
        modelData.data() + header->weightsOffset,
        modelData.data() + header->weightsOffset + header->weightsSize
    );
    
    // Create tensor descriptors
    std::vector<TensorDesc> inputDescs = ParseInputDescs(header);
    std::vector<TensorDesc> outputDescs = ParseOutputDescs(header);
    std::vector<TensorDesc> weightDescs = ParseWeightDescs(header);
    
    // Create model descriptor
    ModelDesc modelDesc = {};
    modelDesc.inputDescs = inputDescs;
    modelDesc.outputDescs = outputDescs;
    modelDesc.weightsData = weights.data();
    modelDesc.operatorGraph = ParseOperatorGraph(header);
    
    // Compile the model
    return CompileModel(modelDesc);
}
```

## Inference Execution

Executing inference involves:

```cpp
HRESULT DirectMLManager::ExecuteModel(const Tensor& inputTensor, Tensor& outputTensor)
{
    // Reset command allocator and open command list
    HRESULT hr = m_commandAllocator->Reset();
    if (FAILED(hr)) return hr;
    
    hr = m_commandList->Reset(m_commandAllocator.Get(), nullptr);
    if (FAILED(hr)) return hr;
    
    // Copy input data to GPU
    D3D12_RESOURCE_BARRIER inputBarrier = {};
    inputBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    inputBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    inputBarrier.Transition.pResource = inputTensor.GetResource();
    inputBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    inputBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    inputBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    
    m_commandList->ResourceBarrier(1, &inputBarrier);
    
    // Bind tensors
    hr = BindTensors(inputTensor, outputTensor);
    if (FAILED(hr)) return hr;
    
    // Record DirectML dispatch
    hr = m_dmlCommandRecorder->RecordDispatch(
        m_commandList.Get(),
        m_dmlCompiledModel.Get(),
        m_dmlBindingTable.Get()
    );
    if (FAILED(hr)) return hr;
    
    // Transition output resource for reading
    D3D12_RESOURCE_BARRIER outputBarrier = {};
    outputBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    outputBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    outputBarrier.Transition.pResource = outputTensor.GetResource();
    outputBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    outputBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    outputBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    
    m_commandList->ResourceBarrier(1, &outputBarrier);
    
    // Close command list and execute
    hr = m_commandList->Close();
    if (FAILED(hr)) return hr;
    
    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
    
    // Wait for execution to complete
    return WaitForGPU();
}
```

## DirectML C# Interoperability

The C# interop layer enables UWP application communication with DirectML:

```csharp
// DirectMLInterop.cs
public class DirectMLInterop
{
    private IntPtr _directMLManagerHandle;
    
    public DirectMLInterop()
    {
        _directMLManagerHandle = DirectMLNative.CreateDirectMLManager();
        if (_directMLManagerHandle == IntPtr.Zero)
        {
            throw new Exception("Failed to create DirectML Manager");
        }
    }
    
    ~DirectMLInterop()
    {
        if (_directMLManagerHandle != IntPtr.Zero)
        {
            DirectMLNative.DestroyDirectMLManager(_directMLManagerHandle);
            _directMLManagerHandle = IntPtr.Zero;
        }
    }
    
    public bool LoadModel(string modelPath)
    {
        return DirectMLNative.LoadModel(_directMLManagerHandle, modelPath);
    }
    
    public bool UnloadModel()
    {
        return DirectMLNative.UnloadModel(_directMLManagerHandle);
    }
    
    public float[] RunInference(float[] inputData, int[] inputShape)
    {
        // Create input tensor descriptor
        TensorDesc inputDesc = new TensorDesc
        {
            DataType = TensorDataType.Float32,
            Sizes = inputShape
        };
        
        // Get expected output shape from the model
        int[] outputShape = GetModelOutputShape();
        if (outputShape == null)
            return null;
        
        // Pre-allocate output array
        int outputElements = 1;
        foreach (int dim in outputShape)
            outputElements *= dim;
        
        float[] outputData = new float[outputElements];
        
        // Call native method
        bool success = DirectMLNative.RunInference(
            _directMLManagerHandle,
            inputData, inputData.Length,
            inputDesc,
            outputData, outputData.Length);
        
        return success ? outputData : null;
    }
    
    private int[] GetModelOutputShape()
    {
        IntPtr shapePtr = DirectMLNative.GetModelOutputShape(_directMLManagerHandle);
        if (shapePtr == IntPtr.Zero)
            return null;
        
        // Parse shape information from native code
        // ...
        
        return shape;
    }
}

// P/Invoke declarations for native methods
internal static class DirectMLNative
{
    [DllImport("DirectMLNative.dll")]
    public static extern IntPtr CreateDirectMLManager();
    
    [DllImport("DirectMLNative.dll")]
    public static extern void DestroyDirectMLManager(IntPtr manager);
    
    [DllImport("DirectMLNative.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool LoadModel(IntPtr manager, [MarshalAs(UnmanagedType.LPWStr)] string modelPath);
    
    [DllImport("DirectMLNative.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool UnloadModel(IntPtr manager);
    
    [DllImport("DirectMLNative.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool RunInference(
        IntPtr manager,
        [In] float[] inputData, int inputDataLength,
        [In] TensorDesc inputDesc,
        [Out] float[] outputData, int outputDataLength);
    
    [DllImport("DirectMLNative.dll")]
    public static extern IntPtr GetModelOutputShape(IntPtr manager);
}

// Supporting structures
[StructLayout(LayoutKind.Sequential)]
public struct TensorDesc
{
    public TensorDataType DataType;
    [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)]
    public int[] Sizes;
    public int SizesCount;
}

public enum TensorDataType
{
    Unknown = 0,
    Float32 = 1,
    Float16 = 2,
    Int32 = 3,
    Int16 = 4,
    Int8 = 5,
    Uint32 = 6,
    Uint16 = 7,
    Uint8 = 8
}
```

## Performance Considerations

Several techniques optimize DirectML performance on Xbox Series S:

### Memory Management
- Use `PlacedResources` instead of `CommittedResources` for fine-grained memory control
- Implement buffer pooling to reduce allocation overhead
- Use upload/readback heaps appropriately for CPU-GPU data transfers

### Command Execution
- Batch multiple operations in a single command list when possible
- Use fences efficiently for synchronization
- Balance work across available command queues

### Shader Optimization
- Use DirectML's built-in shader optimization capabilities
- Consider manually tuning generated shaders for common operations

## Debugging and Profiling

Tools and techniques for debugging DirectML applications:

- Use PIX for Windows for GPU capture and analysis
- Enable DirectML debug layer with `DML_CREATE_DEVICE_FLAG_DEBUG`
- Implement custom performance timers around key operations
- Use Xbox Performance Toolkit for system-wide profiling

## DirectML API Reference

Key DirectML functions and structures used in this implementation:

| API | Description |
|-----|-------------|
| `DMLCreateDevice` | Creates a DirectML device instance |
| `IDMLDevice::CreateOperator` | Creates an operator instance |
| `IDMLDevice::CompileOperator` | Compiles an operator for execution |
| `IDMLCommandRecorder::RecordDispatch` | Records a dispatch command |
| `DML_TENSOR_DESC` | Describes a tensor's dimensions and data type |
| `DML_OPERATOR_DESC` | Base structure for all operator descriptions |
| `DML_BINDING_TABLE_DESC` | Describes bindings for an operator |

## Error Handling

Proper error handling for DirectML operations:

```cpp
// Helper macro for detailed error reporting
#define DML_CHECK_THROW(hr) \
    if (FAILED(hr)) { \
        std::stringstream ss; \
        ss << "DirectML error at " << __FILE__ << ":" << __LINE__ << ": 0x" << std::hex << hr; \
        throw std::runtime_error(ss.str()); \
    }

// Example with error handling
HRESULT DirectMLManager::CreateTensor(const TensorDesc& desc, Tensor& tensor)
{
    try {
        tensor = Tensor(desc);
        
        HRESULT hr = tensor.CreateResource(m_d3d12Device.Get());
        DML_CHECK_THROW(hr);
        
        return S_OK;
    }
    catch (const std::exception& e) {
        OutputDebugStringA(e.what());
        return E_FAIL;
    }
}
``` 