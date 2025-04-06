using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media.Imaging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.Storage.Pickers;

namespace XboxMLApp
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
                    new Uri("ms-appx:///Models/resnet18.dml"));
                
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
            
            // For WinUI 3, we need to initialize the object with a window handle
            var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(App.Current.Windows[0]);
            WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);
            
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
            // In a real implementation, this would:
            // 1. Load and resize image to 224x224 (standard for many models)
            // 2. Convert to RGB float values normalized to [0,1]
            // 3. Apply any model-specific preprocessing
            
            // For this example, we'll return a dummy tensor of the right size
            return new float[3 * 224 * 224]; // RGB image data
        }
        
        private async Task<string[]> LoadClassLabelsAsync()
        {
            try
            {
                // Load ImageNet class labels
                StorageFile labelsFile = await StorageFile.GetFileFromApplicationUriAsync(
                    new Uri("ms-appx:///Models/imagenet_labels.txt"));
                
                var lines = await FileIO.ReadLinesAsync(labelsFile);
                return lines.ToArray();
            }
            catch
            {
                // If we can't load labels, return dummy labels
                return Enumerable.Range(0, 1000).Select(i => $"Class {i}").ToArray();
            }
        }
        
        private List<(string Label, float Confidence)> GetTopNResults(float[] outputs, string[] labels, int n)
        {
            // Find indices of top N values
            return outputs
                .Select((value, index) => (Label: index < labels.Length ? labels[index] : $"Class {index}", Confidence: value))
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
            
            // For WinUI 3, we need to set the XamlRoot
            dialog.XamlRoot = this.XamlRoot;
            
            await dialog.ShowAsync();
        }
    }
} 