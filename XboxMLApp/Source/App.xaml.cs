using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Navigation;
using System;

namespace XboxMLApp
{
    public partial class App : Application
    {
        private Window m_window;

        public App()
        {
            this.InitializeComponent();
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