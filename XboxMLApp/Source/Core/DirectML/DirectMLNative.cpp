#include "DirectMLManager.h"
#include <memory>

// Export functions with C calling convention to be callable from C#
extern "C" {

    __declspec(dllexport) void* __stdcall CreateDirectMLManager()
    {
        try
        {
            auto manager = new DirectMLManager();
            if (FAILED(manager->Initialize()))
            {
                delete manager;
                return nullptr;
            }
            return manager;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    __declspec(dllexport) void __stdcall DestroyDirectMLManager(void* managerPtr)
    {
        if (managerPtr)
        {
            auto manager = static_cast<DirectMLManager*>(managerPtr);
            delete manager;
        }
    }

    __declspec(dllexport) bool __stdcall LoadModel(void* managerPtr, const wchar_t* modelPath)
    {
        if (!managerPtr || !modelPath)
            return false;

        auto manager = static_cast<DirectMLManager*>(managerPtr);
        return SUCCEEDED(manager->LoadModelFromFile(std::wstring(modelPath)));
    }

    __declspec(dllexport) bool __stdcall RunInference(
        void* managerPtr,
        const float* inputData, int inputDataLength,
        float* outputData, int outputDataLength)
    {
        if (!managerPtr || !inputData || !outputData || inputDataLength <= 0 || outputDataLength <= 0)
            return false;

        auto manager = static_cast<DirectMLManager*>(managerPtr);
        return SUCCEEDED(manager->RunInference(inputData, inputDataLength, outputData, outputDataLength));
    }

} 