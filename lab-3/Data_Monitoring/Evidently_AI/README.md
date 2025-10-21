# F1 Data Drift Lab using Evidently AI 🏎️

## Overview
This repository demonstrates **data drift detection** using the Formula 1 World Championship dataset (1950–2020). The lab splits the dataset into **reference** and **production** subsets to simulate changes in driver, constructor, and circuit performance over time.  

**Evidently AI** is used to generate data drift and data summary reports, which help understand how feature distributions change over different seasons.

---

## Dataset
- **Source:** [F1 World Championship 1950–2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)  
- **CSV Files Required:**  
  - `results.csv`  
  - `races.csv`  
- **Selected Columns for Lab:**  
  - Numerical: `position`, `points`, `grid`  
  - Categorical: `driverId`, `constructorId`, `circuitId`  
  - Temporal: `year` (from `races.csv`)


---
## Requirements
- Python 3.9+  
- pandas, numpy  
- evidently (`pip install evidently`)  

---

## Setup and Usage


1. Clone the repository:
```bash
git clone https://github.com/samhita-kolluri/mlops-labs.git
cd lab-3/Data_Monitoring/Evidently_AI
````

2. Download `results.csv` and `races.csv` from Kaggle and place them in the `data` folder.

3. Open `Lab1_F1.ipynb` and run the cells.  
   - Packages are installed via `%pip install` within the notebook.

4. Optional: View interactive reports in Evidently Cloud Workspace by setting your token.

---

## Lab Steps

1. Load and merge CSVs (`results.csv` + `races.csv`) to get `year` and `circuitId`.
2. Select minimal relevant columns (numerical, categorical, and temporal).
3. Split the dataset by `year` to simulate reference and production data.
4. Optionally simulate small artificial drift in `points` and `grid`.
5. Define schema using Evidently AI `DataDefinition`.
6. Generate data drift and summary reports using `DataDriftPreset` and `DataSummaryPreset`.
7. Analyze which features drifted between older and recent seasons.

---

## Key Points

* Year-based split simulates realistic temporal drift.
* Useful for **sports analytics**, **fantasy sports**, or **predictive modeling** where player/team performance changes over time.

