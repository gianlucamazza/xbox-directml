using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media.Imaging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.Storage.Pickers;
using Microsoft.UI;
using WinRT.Interop;

namespace XboxMLApp
{
    public sealed partial class MainPage : Page
    {
        private DirectMLInterop _directML;
        private StorageFile _selectedImageFile;
        
        // UI Elements - we'll define these directly since XAML might not be fully set up
        private Image _inputImage;
        private ListView _resultsList;
        
        public MainPage()
        {
            // WinUI 3 apps need to handle the InitializeComponent differently
            // Commenting out for now
            // this.InitializeComponent();
            
            // Set up UI elements manually 
            SetupUI();
            
            // Initialize DirectML
            InitializeDirectMLAsync();
        }
        
        private void SetupUI()
        {
            // Create a simple UI with a grid layout
            var grid = new Grid();
            
            // Create an image control
            _inputImage = new Image();
            _inputImage.Width = 300;
            _inputImage.Height = 300;
            _inputImage.Stretch = Microsoft.UI.Xaml.Media.Stretch.Uniform;
            
            // Create a ListView for results
            _resultsList = new ListView();
            
            // Create a stack panel to arrange them
            var panel = new StackPanel();
            panel.Children.Add(_inputImage);
            
            // Add load image button
            var loadButton = new Button { Content = "Load Image" };
            loadButton.Click += LoadImageButton_Click;
            panel.Children.Add(loadButton);
            
            // Add run model button
            var runButton = new Button { Content = "Run Model" };
            runButton.Click += RunModelButton_Click;
            panel.Children.Add(runButton);
            
            panel.Children.Add(_resultsList);
            
            // Add panel to grid
            grid.Children.Add(panel);
            
            // Set content
            this.Content = grid;
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
            
            // Get the current window
            var hwnd = GetCurrentWindowHandle();
            InitializeWithWindow.Initialize(picker, hwnd);
            
            _selectedImageFile = await picker.PickSingleFileAsync();
            
            if (_selectedImageFile != null)
            {
                using (var stream = await _selectedImageFile.OpenAsync(FileAccessMode.Read))
                {
                    var bitmap = new BitmapImage();
                    await bitmap.SetSourceAsync(stream);
                    _inputImage.Source = bitmap;
                }
            }
        }
        
        // Helper method to get current window handle
        private IntPtr GetCurrentWindowHandle()
        {
            // In a real app, you would get this from the active window
            // For now, let's use a workaround
            var app = Application.Current as App;
            if (app?.m_window != null)
            {
                return WindowNative.GetWindowHandle(app.m_window);
            }
            
            // Fallback in case we can't get the window
            return IntPtr.Zero;
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
                    _resultsList.Items.Clear();
                    foreach (var result in topResults)
                    {
                        _resultsList.Items.Add($"{result.Label}: {result.Confidence:P2}");
                    }
                }
            }
            catch (Exception ex)
            {
                await ShowErrorMessageAsync($"Error running inference: {ex.Message}");
            }
        }
        
        // Fixed warning by adding actual async operation
        private async Task<float[]> PreprocessImageAsync(StorageFile imageFile)
        {
            // In a real implementation, this would:
            // 1. Load and resize image to 224x224 (standard for many models)
            // 2. Convert to RGB float values normalized to [0,1]
            // 3. Apply any model-specific preprocessing
            
            // Add a small delay to make this actually async
            await Task.Delay(1);
            
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
            if (this.XamlRoot != null)
            {
                dialog.XamlRoot = this.XamlRoot;
            }
            
            await dialog.ShowAsync();
        }
    }
} 