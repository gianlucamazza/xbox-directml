# Xbox DirectML Application

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/gianlucamazza/xbox-directml/xbox-directml-build.yml)

This repository contains a DirectML-based machine learning application for Xbox Series S/X, demonstrating how to run ML inference on Xbox consoles using DirectML.

## Project Overview

The application showcases:

- DirectML integration for machine learning on Xbox
- UWP application deployment to Xbox Series S/X
- Image classification using ResNet-18
- Cross-platform development workflow (macOS/Windows)

## Repository Structure

- **XboxMLApp/**: Main application source code and project files
  - **Source/**: C# and C++ application code
  - **Models/**: ML model files and guides
  - **Documentation/**: Detailed guides and implementation notes

## Build & Deployment

This project uses GitHub Actions for automated building:

1. Changes are committed to the repository
2. GitHub Actions automatically compiles the UWP application on Windows runners
3. Compiled packages can be downloaded from workflow artifacts
4. Packages can be deployed to Xbox consoles in Developer Mode

For detailed information on deployment, see:
- [Windows Deployment Guide](XboxMLApp/WINDOWS_DEPLOYMENT_GUIDE.md)
- [CI/CD Build Process](.github/README-CI.md)

## Development

You can develop this application on:

1. **macOS**: Using VS Code with C# Dev Kit for code editing
2. **Windows**: Using Visual Studio 2022 for full build and deployment

For setup instructions, see:
- [Cross-Platform Build Notes](XboxMLApp/CROSS_PLATFORM_BUILD_NOTES.md)
- [Model Setup Guide](XboxMLApp/Models/MODEL_SETUP_GUIDE.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft DirectML team for DirectML technology
- Xbox development community 