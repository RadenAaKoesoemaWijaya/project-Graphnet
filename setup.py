from setuptools import setup, find_packages

with open("README_COMPLETE.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="astina",
    version="1.0.0",
    author="ASTINA Team",
    author_email="support@astina.ai",
    description="Analisis Sistem Transaksi Identifikasi Nilai Anomali using Hybrid Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/astina/astina",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch[cuda]",
        ],
    },
    entry_points={
        "console_scripts": [
            "astina=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.md", "*.txt"],
    },
    keywords="transaction fraud detection machine learning graph neural networks isolation forest autoencoder finance",
    project_urls={
        "Bug Reports": "https://github.com/astina/astina/issues",
        "Source": "https://github.com/astina/astina",
        "Documentation": "https://docs.astina.ai",
    },
)
