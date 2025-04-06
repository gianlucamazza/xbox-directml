using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Navigation;
using System;

namespace XboxMLApp
{
    public partial class App : Application
    {
        // Make m_window public so it can be accessed from MainPage
        public Window m_window;

        // Add entry point for UWP application
        [System.STAThreadAttribute]
        public static void Main(string[] args)
        {
            global::WinRT.ComWrappersSupport.InitializeComWrappers();
            Microsoft.UI.Xaml.Application.Start((p) => { new App(); });
        }

        public App()
        {
            // WinUI 3 apps need to handle the InitializeComponent differently
            // since it's not automatically generated for App.xaml in some cases
            // Commenting out the call for now
            // this.InitializeComponent();
        }

        protected override void OnLaunched(Microsoft.UI.Xaml.LaunchActivatedEventArgs args)
        {
            m_window = new Window();
            m_window.Activate();

            // Create a Frame to act as the navigation context
            Frame rootFrame = new Frame();

            // Place the frame in the current Window
            m_window.Content = rootFrame;

            // Navigate to the first page
            rootFrame.Navigate(typeof(MainPage));

            // Ensure the current window is active
            m_window.Activate();
        }
    }
} 