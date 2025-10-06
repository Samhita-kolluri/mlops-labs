# Formula 1 Driver Performance Clustering with Airflow 🏎️

## Overview

This project demonstrates an **MLOps workflow using Apache Airflow** to automate the clustering of Formula 1 driver performance data.
We use **K-Means clustering** to group drivers based on key numeric metrics (`points`, `position`, `wins`) across seasons.

The Airflow DAG handles the full pipeline:

1. Loading and serializing data
2. Preprocessing and scaling
3. Building and saving a K-Means model
4. Using the elbow method to determine the optimal number of clusters
5. Making predictions on test data

---

## Dataset

* **Source:** [Formula 1 World Championship 1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020/data)
* **Files used:**

  * `file.csv` – driver performance data for clustering
  * `test.csv` – sample data for testing model predictions
* **Key features for clustering:** `points`, `position`, `wins`

---

## Project Structure


```
Airflow_Labs
└── Lab_1
    ├── README.md
    ├── config
    │   └── airflow.cfg
    ├── dags
    │   ├── airflow.py              # Airflow DAG definition         
    │   ├── data
    │   │   ├── file.csv            # Main dataset
    │   │   └── test.csv            # Sample test dataset
    │   └── model                   # Saved KMeans model output 
    │   └── src
    │       ├── __init__.py
    │       └── lab.py              # Functions: load, preprocess, train, predict
    ├── docker-compose.yaml         # Airflow Docker setup
    └── setup.sh   

```
----

## Prerequisites

### Local

* **Docker Desktop** running
* Allocate **≥ 4GB RAM** (8GB recommended)
* **Python 3.10+** (if testing scripts locally)

### Airflow

* Airflow 3.x with **CeleryExecutor**
* Docker Compose configuration provided in `docker-compose.yaml`

---

## Setup Instructions

1. **Clone repository**

```bash
git clone https://github.com/Samhita-kolluri/mlops-labs
cd mlops-labs/Labs/Airflow_Labs/Lab_1
````

2. **Start Airflow via Docker Compose**

```bash
bash setup.sh
docker-compose up airflow-init
docker-compose up -d
```

3. **Check Airflow UI**

* Open your browser: [http://localhost:8080](http://localhost:8080)
* DAG: `Airflow_Lab1` should appear

4. **Run DAG**

* Trigger manually in the UI or use CLI:

```bash
docker-compose run airflow-cli dags trigger Airflow_Lab1
```


## How it Works

1. **Load Data**

   * `load_data()` reads `file.csv` and serializes it for Airflow XCom transfer.

2. **Data Preprocessing**

   * `data_preprocessing()` scales `points`, `position`, and `wins` features for clustering.

3. **Build & Save Model**

   * `build_save_model()` runs K-Means for k=1 to 14 clusters
   * Saves the trained model to `model/model.sav`

4. **Elbow & Prediction**

   * `load_model_elbow()` determines the optimal number of clusters using the elbow method
   * Makes a sample prediction on `test.csv`

---

![image](..\assets\2-flow.png)

Author: **Samhita Kolluri** 🏁