"""
This file is used to build the project and required dependencies and install it in the current environment.
"""

from setuptools import find_packages, setup

setup(
    name="zebrafish-procrustes-analysis",
    version="0.1.0",
    description="A tool for analyzing zebrafish movement using Procrustes analysis on video data.",
    url="https://github.com/your-username/zebrafish-procrustes-analysis",
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(exclude=["test", "tests", ".github", ".venv"]),
    py_modules=["main"],
    install_requires=[
        "matplotlib",
        "numpy",
        "opencv-python",
        "pandas",
        "pytest",
        "scipy",
        "tox",
        "tqdm",
        "h5py",
    ],
    entry_points={
        "console_scripts": [
            "procrustes-analysis = main:main",
            "zebrafish-analysis = main:main",
        ]
    },
    extras_require={
        "test": ["pytest"],
        "format": ["ruff", "ty"],
        "lint": ["ruff", "ty"],
    },
    python_requires=">=3.8",
)
