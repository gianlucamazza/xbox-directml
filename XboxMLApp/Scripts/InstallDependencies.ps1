# Script to install VCLibs on Xbox
# This is often required for UWP apps developed with Visual Studio

param(
    [Parameter(Mandatory=$true)]
    [string]$XboxIpAddress
)

Write-Host "Attempting to install VCLibs on Xbox at $XboxIpAddress..."

# URL for required VCLibs
$vcLibsUrl = "https://aka.ms/Microsoft.VCLibs.arm64.14.00.Desktop.appx"
$outFile = "Microsoft.VCLibs.arm64.14.00.Desktop.appx"

# Download VCLibs
Write-Host "Downloading VCLibs..."
Invoke-WebRequest -Uri $vcLibsUrl -OutFile $outFile

# Install VCLibs on Xbox (requires WinAppDeployCmd)
Write-Host "Installing VCLibs on Xbox..."
& WinAppDeployCmd install -File $outFile -IP $XboxIpAddress -ConnectionType Remote

Write-Host "VCLibs installation process completed."
Write-Host "Now try reinstalling the main application." 