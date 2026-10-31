param(
    [string]$Repository = (Join-Path $PSScriptRoot '../../FastSAM'),
    [ValidateSet('x', 's')][string]$Variant = 'x',
    [switch]$SkipDependencies
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$fastsamRepository = (Resolve-Path -LiteralPath $Repository).Path
if (-not (Test-Path -LiteralPath (Join-Path $fastsamRepository 'fastsam/model.py'))) {
    throw "Not a FastSAM clone: $fastsamRepository"
}
$environmentRoot = Join-Path $projectRoot 'target/fastsam-env'
$python = Join-Path $environmentRoot 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & python -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the FastSAM Python environment' }
}
function Invoke-SetupPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "FastSAM setup failed: python $($Arguments -join ' ')" }
}
if (-not $SkipDependencies) {
    Invoke-SetupPython -m pip --isolated install --upgrade pip --index-url https://pypi.org/simple
    # CPU PyTorch is sufficient for the one-time export. Native ToolTrace uses
    # ONNX Runtime / DirectML, with CPU fallback when DirectML is unavailable.
    Invoke-SetupPython -m pip --isolated install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
    Invoke-SetupPython -m pip --isolated install -r (Join-Path $fastsamRepository 'requirements.txt') numpy==1.26.4 opencv-python==4.11.0.86 onnx==1.17.0 psutil thop gradio_client==0.2.7 --index-url https://pypi.org/simple
}
# Make the cloned fastsam and its bundled ultralytics importable in this venv.
Set-Content -LiteralPath (Join-Path $environmentRoot 'Lib/site-packages/fastsam-clone.pth') -Value $fastsamRepository -Encoding ascii
$weightsDirectory = Join-Path $projectRoot 'target/fastsam-weights'
New-Item -ItemType Directory -Force -Path $weightsDirectory | Out-Null
$checkpoint = Join-Path $weightsDirectory "FastSAM-$Variant.pt"
if (-not (Test-Path -LiteralPath $checkpoint)) {
    & curl.exe -L --fail --retry 2 --output $checkpoint "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-$Variant.pt"
    if ($LASTEXITCODE -ne 0) { throw 'Could not download the official FastSAM checkpoint' }
}
$env:PYTHONDONTWRITEBYTECODE = '1'
Invoke-SetupPython (Join-Path $PSScriptRoot 'export_fastsam.py') --repository $fastsamRepository --weights $checkpoint --output (Join-Path $projectRoot "models/FastSAM-$Variant.onnx")
& $python -m pip freeze | Set-Content -LiteralPath (Join-Path $projectRoot 'target/fastsam-environment.txt') -Encoding utf8
Write-Output 'FastSAM is installed. ToolTrace discovers the ONNX model automatically; no Python process is needed for tracing.'
