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

# Suppress Polars CPU feature check warnings
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

def check_requirements():
    """Periksa apakah semua paket yang dibutuhkan sudah terpasang."""
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
        try:
            __import__(package_import_map.get(package, package.replace('-', '_')))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Ada paket yang belum terpasang:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Instal dengan: pip install -r requirements.txt")
        return False
    
    print("✅ Semua paket yang dibutuhkan sudah terpasang")
    return True

def main():
    """Fungsi utama untuk menjalankan aplikasi."""
    print("🛡️ ASTINA - Analisis Sistem Transaksi Identifikasi Nilai Anomali")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("studies", exist_ok=True)
    
    print("🚀 Menjalankan aplikasi...")
    print("🌐 Membuka di browser pada http://localhost:8501")
    print("⏹️  Tekan Ctrl+C untuk menghentikan aplikasi")
    print()
    
    try:
        # Run streamlit app with 3GiB upload capacity
        cmd = [
            sys.executable, "-m", "streamlit", "run", "main.py",
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
