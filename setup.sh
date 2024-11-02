#!/bin/bash

# 1. Unzip research_papers.zip into papers/research_papers
echo "Preparing to unzip research_papers.zip..."
if [ ! -d "papers" ]; then
  echo "Creating papers/research_papers directory..."
  mkdir -p papers
else
  echo "Directory papers/research_papers already exists."
fi

echo "Unzipping research_papers.zip into papers/research_papers..."
unzip -j research_papers.zip -d papers/research_papers || { echo "Unzip failed"; exit 1; }

# 2. Create a Python virtual environment and install packages from requirements.txt
echo "Setting up Python virtual environment..."
python3 -m venv env || { echo "Failed to create virtual environment"; exit 1; }

echo "Activating virtual environment and installing packages..."
source env/bin/activate
pip install -r requirements.txt || { echo "Failed to install packages"; exit 1; }

# 3. Add marker-pdf into PATH
echo "Adding path_to_marker to PATH..."
path_to_marker=$(pip show marker-pdf | grep "Location" | awk '{print $2}')
export PATH="$path_to_marker:$PATH"

# Confirming PATH update
echo "Package path added to PATH: $path_to_marker"
echo "Script completed successfully!"