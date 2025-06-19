"""
Setup script for CausalKit.
"""

from setuptools import setup, find_packages

setup(
    name="causalkit",
    version="0.1.0",
    description="A Python toolkit for causal inference and experimentation",
    author="CausalKit Team",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/CausalKit",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "plotly>=5.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
)
