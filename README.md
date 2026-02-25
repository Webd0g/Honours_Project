# Honours Project — [Astrophysics Project]

## 📌 Overview
This repository contains the code, notebooks, data structures, and documentation for my Honours research project in [your field, e.g., cosmology/astronomy/physics]. The project explores:

- [Brief description of your research question]
- [Key goals or hypotheses]
- [Methods or datasets used]

The repository is organised to support a clean, reproducible scientific workflow using Python, virtual environments, Jupyter notebooks, and Git.

---

## 🗂 Project Structure
Honours_Project/ │ ├── .venv/                     # Virtual environment (ignored by Git) ├── src/                       # Python source code ├── notebooks/                 # Jupyter notebooks for analysis ├── data/                      # Raw and processed data (ignored by Git) ├── figures/                   # Generated plots and figures ├── images/                    # Images used in documentation ├── Thesis/                    # Thesis writing and LaTeX files ├── Classes/                   # Coursework or reference material ├── requirements.txt           # Python dependencies └── README.md                  # Project documentation

---

## 🐍 Python Environment Setup

### 1. Create and activate the virtual environment
```powershell
python -m venv .venv
.\.honours_project\Scripts\activate

---

### 2. Install Dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

---

### 3. Register the environment as a Jupyter kernel
python -m ipykernel install --user --name honours-env

---

## 📓 Working With Notebooks
All notebooks are stored in the notebooks/ directory.
To ensure reproducibility, select the honours-env kernel in Jupyter or VS Code before running any notebook.

---

## 🧪 Running Code
Python Scripts live in the src/ directory 
Example: 
python src/example_script.py

