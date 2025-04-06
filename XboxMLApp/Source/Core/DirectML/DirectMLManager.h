#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <wrl/client.h>
#include <vector>
#include <string>

using Microsoft::WRL::ComPtr;

// Forward declarations
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