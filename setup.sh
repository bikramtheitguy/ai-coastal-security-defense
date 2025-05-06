#!/bin/bash

echo "🔧 Setting up AI Coastal Security Project..."

# Create folders
mkdir -p data notebooks scripts models dashboard

# Add README
echo "# AI-Driven Coastal Security

Project initialized." > README.md

# Add requirements.txt
cat <<EOL > requirements.txt
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
jupyter==1.0.0
notebook==7.1.3
EOL

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
echo "🚀 Launching Jupyter Notebook..."
jupyter notebook notebooks/
