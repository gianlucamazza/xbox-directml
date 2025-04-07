# CI/CD for Xbox DirectML App

This repository includes a GitHub Actions workflow configured to automatically build the DirectML application for Xbox Series S/X.

## Workflow Features

- **Automatic compilation** with each push to the main branch
- **ARM64 build** optimized for Xbox Series S/X
- **MSIX package generation** ready for deployment
- **DirectML model support** with automatic placeholder
- **Deployment instructions** generated with each build

## How to Use the Workflow

### Manual Execution

1. Go to the "Actions" tab of the GitHub repository
2. Select the "Xbox DirectML App Build" workflow from the left list
3. Click the "Run workflow" button and confirm execution
4. Wait for the build to complete (approximately 5-10 minutes)

### Downloading Artifacts

After the build is complete:

1. Open the completed workflow run
2. Scroll down to the "Artifacts" section at the bottom of the page
3. Download the following artifacts:
   - **XboxMLApp-Deployment**: Complete ZIP package with all necessary files
   - **Deployment-Instructions**: Quick deployment guide

### Deployment to Xbox

1. Extract the XboxMLApp-Deployment.zip package on a Windows PC
2. Follow the instructions in the included WINDOWS_DEPLOYMENT_GUIDE.md
3. Use Xbox Device Portal (https://[xbox-ip-address]) to upload the MSIX package

## Configuration and Customization

The workflow is defined in the `.github/workflows/xbox-directml-build.yml` file. You can modify it to:

- Change trigger conditions (branches or paths)
- Add test or validation steps
- Configure package signing with custom certificates
- Modify MSBuild build options

## Requirements for Local Development

If you wish to build locally instead of using GitHub Actions:

- Windows 10/11
- Visual Studio 2022 with UWP and Xbox workloads
- .NET 6.0 SDK
- Windows 10 SDK (10.0.19041.0)

## Known Issues

- The automatically generated certificate expires after 30 days
- Developer Mode must be enabled on the Xbox for installation
- The DirectML model must be provided separately (see MODEL_SETUP_GUIDE.md) 