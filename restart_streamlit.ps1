Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$streamlitExe = Join-Path $projectRoot '.venv\Scripts\streamlit.exe'
$port = 8505

function Stop-ProjectProcesses {
    $allProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
    if (-not $allProcesses) {
        Write-Host "Tidak ada proses yang dapat diperiksa." -ForegroundColor Green
        return
    }

    foreach ($proc in $allProcesses) {
        $procId = $proc.ProcessId
        $procName = [string]($proc.Name)
        $cmdLine = [string]($proc.CommandLine)

        if (-not $procId) { continue }
        $targetProcess = ($procName -match 'python|streamlit' -or $cmdLine -match 'streamlit|main.py') -and (
            $cmdLine -match 'project-Graphnet-main' -or
            $cmdLine -match 'streamlit' -or
            $cmdLine -match 'main.py'
        )

        if ($targetProcess) {
            try {
                Stop-Process -Id $procId -Force -ErrorAction Stop
                Write-Host ("Stopped PID {0}: {1}" -f $procId, $procName) -ForegroundColor DarkYellow
            } catch {
                Write-Host ("Could not stop PID {0}: {1}" -f $procId, $_.Exception.Message) -ForegroundColor Red
            }
        }
    }
}

Write-Host "[1/4] Membersihkan proses Streamlit/Python lama..." -ForegroundColor Yellow
Stop-ProjectProcesses

Write-Host "[2/4] Membebaskan port $port..." -ForegroundColor Yellow
$portConnections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
if ($portConnections) {
    foreach ($conn in $portConnections) {
        $ownerPid = $conn.OwningProcess
        if (-not $ownerPid) { continue }
        try {
            Stop-Process -Id $ownerPid -Force -ErrorAction Stop
            Write-Host ("Port {0} dibebaskan dari PID {1}" -f $port, $ownerPid) -ForegroundColor DarkYellow
        } catch {
            Write-Host ("Gagal membebaskan port {0} dari PID {1}" -f $port, $ownerPid) -ForegroundColor Red
        }
    }
} else {
    Write-Host "Port $port masih tersedia." -ForegroundColor Green
}

Write-Host "[3/4] Validasi import state_manager..." -ForegroundColor Yellow
& $venvPython -c "import state_manager; print('IMPORT_OK', hasattr(state_manager, 'navigate_to_page'))"
if ($LASTEXITCODE -ne 0) {
    throw "Import state_manager gagal. Periksa environment virtual dan file project."
}

Write-Host "[4/4] Menjalankan Streamlit..." -ForegroundColor Yellow
& $streamlitExe run main.py --server.address 127.0.0.1 --server.port $port --server.headless true
