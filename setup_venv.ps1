Param(
  [string]$PythonTag = "3.13",
  [string]$VenvDir = ".venv",
  [switch]$Force
)

$ErrorActionPreference = "Stop"

function Get-PyLauncherPath {
  try {
    $null = & py -0p 2>$null
    return $true
  } catch {
    return $false
  }
}

function Find-PythonByTag([string]$Tag) {
  try {
    $out = & py -$Tag -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
  } catch {}
  return $null
}

Write-Host "== ASTINA Environment Setup ==" -ForegroundColor Cyan
Write-Host "Target Python: $PythonTag" -ForegroundColor Cyan

if (-not (Get-PyLauncherPath)) {
  Write-Host "ERROR: Python Launcher 'py' tidak ditemukan." -ForegroundColor Red
  Write-Host "Install Python 3.13 dari https://www.python.org/downloads/ dan pastikan opsi 'Install launcher for all users' dicentang." -ForegroundColor Yellow
  exit 1
}

$pythonExe = Find-PythonByTag $PythonTag
if (-not $pythonExe) {
  Write-Host "ERROR: Python $PythonTag tidak terdeteksi oleh 'py' launcher." -ForegroundColor Red
  Write-Host "Cek versi python yang tersedia dengan: py -0p" -ForegroundColor Yellow
  Write-Host "Install Python 3.13 dari https://www.python.org/downloads/ lalu jalankan script ini lagi." -ForegroundColor Yellow
  exit 1
}

Write-Host "Using: $pythonExe" -ForegroundColor Green

function Stop-VenvProcesses {
  $fullVenv = [System.IO.Path]::GetFullPath($VenvDir)
  $allProcs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
  if (-not $allProcs) { return }
  foreach ($p in $allProcs) {
    if (-not $p.ProcessId -or $p.ProcessId -eq $PID) { continue }
    $pCmd = [string]($p.CommandLine)
    $pPath = [string]($p.ExecutablePath)
    if ($pCmd -like "*$fullVenv*" -or $pPath -like "*$fullVenv*") {
      try {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
        Write-Host "Menghentikan proses yang mengunci venv: PID $($p.ProcessId) ($($p.Name))" -ForegroundColor Yellow
      } catch {}
    }
  }
}

if (Test-Path $VenvDir) {
  if ($Force) {
    Write-Host "Removing existing venv: $VenvDir" -ForegroundColor Yellow
    Stop-VenvProcesses
    Remove-Item -Recurse -Force $VenvDir
  } else {
    Write-Host "Venv already exists: $VenvDir (use -Force to recreate)" -ForegroundColor Yellow
  }
}

if (-not (Test-Path $VenvDir)) {
  Stop-VenvProcesses
  & py -$PythonTag -m venv $VenvDir
}

$activatePath = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
  Write-Host "ERROR: Aktivasi venv tidak ditemukan: $activatePath" -ForegroundColor Red
  exit 1
}

Write-Host "Upgrading pip tooling..." -ForegroundColor Cyan
& (Join-Path $VenvDir "Scripts\python.exe") -m pip install -U pip setuptools wheel

Write-Host "Installing requirements..." -ForegroundColor Cyan
& (Join-Path $VenvDir "Scripts\pip.exe") install --prefer-binary -r requirements.txt

Write-Host "DONE." -ForegroundColor Green
Write-Host "Aktifkan venv dengan: .\$VenvDir\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Jalankan app dengan: streamlit run main.py" -ForegroundColor Green
