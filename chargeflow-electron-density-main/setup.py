"""
Setup configuration for Charged Electron Density Prediction package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "torchdiffeq>=0.2.0",
        "tqdm>=4.60.0",
    ]

setup(
    name="chargeflow",
    version="1.0.0",
    author="Tri Minh Nguyen",
    author_email="",
    description="ChargeFlow code for charge-conditioned electron density prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ngminhtri0394/chargeflow-electron-density",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "logging": [
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "edp-train=scripts.train:main",
            "edp-predict=scripts.predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
