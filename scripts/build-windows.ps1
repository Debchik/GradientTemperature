Param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\.."),
    [string]$VenvPath = "$ProjectRoot\.venv",
    [string]$PyInstallerExe = "$VenvPath\Scripts\pyinstaller.exe"
)

if (-not (Test-Path $PyInstallerExe)) {
    Write-Error "PyInstaller not found at $PyInstallerExe. Activate the virtual environment or point PyInstallerExe to a valid binary."
    exit 1
}

Push-Location $ProjectRoot
& $PyInstallerExe --noconfirm ".\GradientTemperature.spec"
Pop-Location
