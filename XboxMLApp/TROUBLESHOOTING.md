# Troubleshooting error 0x800700d8

If you are encountering the error with code 0x800700d8 when launching the application, follow these steps:

## Cause of the error

The error 0x800700d8 (with message "The version you have now is out of date") typically indicates one of the following problems:

1. **Missing dependencies** - Some libraries required by the app are not present on the Xbox
2. **Architecture incompatibility** - The app is not correctly compiled for Xbox (ARM64)
3. **Insufficient permissions or capabilities** - The app manifest does not have the necessary permissions
4. **App Framework issues** - Missing frameworks such as VCLibs or .NET Runtime

## Solutions to try

### 1. Install VCLibs

UWP apps often require Visual C++ Runtime Libraries. Run:

```powershell
.\InstallDependencies.ps1 -XboxIpAddress [YOUR-XBOX-IP]
```

### 2. Verify that the Xbox is in developer mode

Make sure your Xbox is in developer mode and that you are logged in with the same account used for development.

### 3. Try reinstalling the app after a restart

1. Restart the Xbox
2. Uninstall the app from the Xbox control panel
3. Reinstall the app via the Device Portal

### 4. Update your Xbox

Make sure your Xbox is updated to the latest operating system version.

### 5. If nothing else works

If you continue to experience problems, try:

1. Building the app on Windows with Visual Studio
2. Creating a signed MSIX package with Visual Studio
3. Installing the MSIX package on the Xbox via the Device Portal

## References

- [Common UWP app errors on Xbox](https://docs.microsoft.com/en-us/windows/uwp/xbox-apps/known-issues)
- [Troubleshooting Xbox Developer Mode](https://docs.microsoft.com/en-us/windows/uwp/xbox-apps/troubleshooting)
- [App Deployment on Xbox](https://docs.microsoft.com/en-us/windows/uwp/xbox-apps/development-environment-setup) 