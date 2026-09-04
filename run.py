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
if sys.platform == "win32":
    import asyncio
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

    print("🚀 Menjalankan aplikasi...")
    print("🌐 Membuka di browser pada http://localhost:8501")
    print("⏹️  Tekan Ctrl+C untuk menghentikan aplikasi")
    print()

    try:
        # Run streamlit app using the resolved Python interpreter
        cmd = [
            PYTHON_EXEC, "-m", "streamlit", "run", "main.py",
            "--server.maxUploadSize=3072",
            "--server.maxMessageSize=3072"
        ]
        subprocess.run(cmd, check=True)
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
