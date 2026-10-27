<#
.SYNOPSIS
    Create the Python environments for this project on a new PC.

.DESCRIPTION
    .venv             Python 3.12, camera apps (main.py, trash_talk.py,
                      music_control.py), from requirements-app.txt.
    .venv-mediapipe   Python 3.10, model training (training_models.py),
                      from requirements.txt.

    Needs the Python launcher (py) with Python 3.12 and, for training,
    Python 3.10 installed from python.org. Running it again is safe: existing
    environments are reused and only the packages are (re)installed.

.EXAMPLE
    .\setup.ps1                 # both environments
    .\setup.ps1 -SkipTraining   # only the camera apps
#>
param(
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    # Run a native command and stop if it fails. Output goes straight to the
    # console so it does not end up in a function's return value.
    & $args[0] $args[1..($args.Count - 1)] | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $($args -join ' ')"
    }
}

function New-Venv($Name, $PythonVersion) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "The Python launcher 'py' was not found. Install Python $PythonVersion from python.org."
    }
    & py "-$PythonVersion" --version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Python $PythonVersion is not installed. Install it from python.org, then run this script again."
    }
    if (-not (Test-Path "$Name\Scripts\python.exe")) {
        Write-Host "Creating $Name with Python $PythonVersion..."
        Invoke-Checked py "-$PythonVersion" -m venv $Name
    } else {
        Write-Host "Reusing existing $Name"
    }
    $python = Join-Path $PSScriptRoot "$Name\Scripts\python.exe"
    Invoke-Checked $python -m pip install --upgrade pip --quiet
    return $python
}

Write-Host "`n== Camera apps environment (.venv) =="
$app = New-Venv ".venv" "3.12"
Invoke-Checked $app -m pip install -r requirements-app.txt
Invoke-Checked $app -c "import cv2, mediapipe, soundcard, winrt.windows.media.control; print('camera apps OK')"

if (-not $SkipTraining) {
    # The pinned training packages need these build steps in this order;
    # see HowToRun.txt for why.
    Write-Host "`n== Training environment (.venv-mediapipe) =="
    $train = New-Venv ".venv-mediapipe" "3.10"
    Invoke-Checked $train -m pip install "Cython<3" "setuptools<81" wheel
    Invoke-Checked $train -m pip install --no-build-isolation "PyYAML==5.4.1"
    Invoke-Checked $train -m pip install -r requirements.txt
    Invoke-Checked $train -m pip install --no-deps "protobuf==3.20.3"
    Invoke-Checked $train -c "import tensorflow, mediapipe_model_maker; print('training OK')"
}

Write-Host "`nDone. Run the apps with:"
Write-Host "    .\.venv\Scripts\python.exe music_control.py"
if (-not $SkipTraining) {
    Write-Host "Train with:"
    Write-Host "    .\.venv-mediapipe\Scripts\python.exe training_models.py"
}
