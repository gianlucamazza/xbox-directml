using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace XboxMLApp
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
            [In] float[] inputData, int inputDataLength,
            [Out] float[] outputData, int outputDataLength);
        
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