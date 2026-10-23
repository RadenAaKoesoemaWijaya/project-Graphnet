#!/usr/bin/env python3
"""
ASTINA - Analisis Sistem Transaksi Identifikasi Nilai Anomali
Skrip utama untuk menjalankan aplikasi
"""

import subprocess
import sys
import os

# Fix for Windows console UnicodeEncodeError with emojis
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# FIX: Windows asyncio ProactorEventLoop ConnectionResetError (Python 3.12+)
# Must be applied BEFORE any asyncio loop starts (including uvicorn's loop).
# ProactorEventLoop has a known bug causing unhandled exceptions when browsers
# disconnect: _ProactorBasePipeTransport._call_connection_lost → [WinError 10054]
# WindowsSelectorEventLoopPolicy is stable for Streamlit's HTTP/WebSocket I/O.
# ─────────────────────────────────────────────────────────────────────────────
if sys.platform == "win32" and sys.version_info < (3, 14):
    import asyncio
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ─────────────────────────────────────────────────────────────────────────────
# FIX: Resolve correct Python interpreter (.venv priority)
# Running `python run.py` with system Python can cause dependency conflicts
# (e.g., scipy 1.18.0 + system torch 2.7.1 → AttributeError: torch.Tensor).
# Always use the project's .venv Python to ensure dependency compatibility.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_python_interpreter() -> str:
    """Return the best Python interpreter: .venv first, then current sys.executable."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Windows venv paths
    venv_candidates = [
        os.path.join(project_root, ".venv", "Scripts", "python.exe"),
        os.path.join(project_root, "venv", "Scripts", "python.exe"),
        # Linux/macOS venv paths (for Docker / Cloud Run)
        os.path.join(project_root, ".venv", "bin", "python"),
        os.path.join(project_root, "venv", "bin", "python"),
    ]
    
    for candidate in venv_candidates:
        if os.path.isfile(candidate):
            return candidate
    
    # Fallback: use current interpreter (already in venv if activated)
    return sys.executable

PYTHON_EXEC = _resolve_python_interpreter()

# Suppress Polars CPU feature check warnings
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"


def check_requirements():
    """Periksa apakah semua paket yang dibutuhkan sudah terpasang (using resolved Python)."""
    required_packages = [
        'streamlit', 'torch', 'torch_geometric', 'pandas',
        'numpy', 'scikit-learn', 'plotly', 'networkx',
        'matplotlib', 'seaborn', 'joblib', 'tqdm', 'optuna'
    ]
    package_import_map = {
        'scikit-learn': 'sklearn',
        'torch_geometric': 'torch_geometric',
    }

    missing_packages = []
    for package in required_packages:
        import_name = package_import_map.get(package, package.replace('-', '_'))
        result = subprocess.run(
            [PYTHON_EXEC, "-c", f"import {import_name}"],
            capture_output=True
        )
        if result.returncode != 0:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Ada paket yang belum terpasang:")
        for package in missing_packages:
            print(f"   - {package}")
        print(f"\n📦 Instal dengan: {PYTHON_EXEC} -m pip install -r requirements.txt")
        return False

    print("✅ Semua paket yang dibutuhkan sudah terpasang")
    return True


def main():
    """Fungsi utama untuk menjalankan aplikasi."""
    print("🛡️ ASTINA - Analisis Sistem Transaksi Identifikasi Nilai Anomali")
    print("=" * 60)

    # Show which Python interpreter is being used
    is_venv = ".venv" in PYTHON_EXEC or "venv" in PYTHON_EXEC
    venv_label = "✅ Virtual Environment (.venv)" if is_venv else "⚠️  System Python (aktifkan .venv untuk hasil optimal)"
    print(f"🐍 Python: {PYTHON_EXEC}")
    print(f"🔧 Environment: {venv_label}")
    print()

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("studies", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    # Configure thread pool and streaming limits to prevent memory exhaustion and crashes
    max_threads = str(min(4, max(1, (os.cpu_count() or 2) // 2)))
    os.environ["POLARS_MAX_THREADS"] = max_threads
    os.environ["RAYON_NUM_THREADS"] = max_threads
    os.environ["OMP_NUM_THREADS"] = max_threads
    os.environ["OPENBLAS_NUM_THREADS"] = max_threads
    os.environ["MKL_NUM_THREADS"] = max_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = max_threads
    os.environ["NUMEXPR_NUM_THREADS"] = max_threads
    os.environ["POLARS_STREAMING_CHUNK_SIZE"] = "50000"

    print("🚀 Menjalankan aplikasi...")
    print("🌐 Membuka di browser pada http://localhost:8501")
    print("⏹️  Tekan Ctrl+C untuk menghentikan aplikasi")
    print()

    try:
        # Run streamlit app using the resolved Python interpreter
        runtime_env = os.environ.copy()
        runtime_env["POLARS_MAX_THREADS"] = max_threads
        runtime_env["RAYON_NUM_THREADS"] = max_threads
        runtime_env["OMP_NUM_THREADS"] = max_threads
        runtime_env["OPENBLAS_NUM_THREADS"] = max_threads
        runtime_env["MKL_NUM_THREADS"] = max_threads
        runtime_env["VECLIB_MAXIMUM_THREADS"] = max_threads
        runtime_env["NUMEXPR_NUM_THREADS"] = max_threads
        runtime_env["POLARS_STREAMING_CHUNK_SIZE"] = "50000"
        project_root = os.path.dirname(os.path.abspath(__file__))
        existing_pythonpath = runtime_env.get("PYTHONPATH", "")
        runtime_env["PYTHONPATH"] = os.pathsep.join(
            path for path in (project_root, existing_pythonpath) if path
        )
        cmd = [
            PYTHON_EXEC, "-m", "streamlit", "run", "main.py",
            "--server.maxUploadSize=3072",
            "--server.maxMessageSize=3072",
            "--server.fileWatcherType=none",
            "--server.websocketPingInterval=60",
            "--server.websocketPingTimeout=120"
        ]
        subprocess.run(cmd, check=True, env=runtime_env)
    except KeyboardInterrupt:
        print("\n⏹️  Aplikasi dihentikan oleh pengguna")
    except subprocess.CalledProcessError as e:
        print(f"❌ Gagal menjalankan aplikasi: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Terjadi kesalahan tak terduga: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
