#!/bin/bash

# Lead-Lag Stock Analysis System - Installation Script
# This script sets up the Python environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Lead-Lag Stock Analysis System - Installation"
echo "================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Found Python $PYTHON_VERSION"

# Check if version is 3.8 or higher
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ All packages installed successfully"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Create necessary directories
echo "📁 Creating output directories..."
mkdir -p output
mkdir -p plots
mkdir -p cache
echo "✅ Directories created"

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import sklearn
import streamlit as st
print('✅ All packages imported successfully')
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Run example analysis: python example_usage.py"
echo "   3. Start web interface: streamlit run streamlit_app.py"
echo "   4. Or run CLI analysis: python main.py"
echo ""
echo "📚 For more information, see README.md"
echo "⚠️  Remember: This is for educational purposes only!"
echo ""
echo "🔧 To activate the environment in the future, run:"
echo "   source venv/bin/activate"