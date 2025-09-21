# 🌍 PM2.5 Air Quality Prediction with Enhanced Bidirectional LSTM

## 📌 Overview
This project implements an **Enhanced Bidirectional LSTM model** to predict **PM2.5 air pollutant concentration** from meteorological and temporal data.  
The main objective is to achieve an **aggressive Kaggle leaderboard score target of 2000 RMSE** by leveraging **focused feature engineering, robust preprocessing, and sequence modeling**.

---

## 🚀 Features
- **Focused Feature Engineering**  
  - Temporal: hour, day of week, month, season, day of year (with cyclical encoding).  
  - Weather: temperature, dew point, wind, pressure, and interactions.  
  - PM2.5 history: lag features, rolling mean/std, trends.  

- **Preprocessing**  
  - Robust missing value handling (forward/backward fill, interpolation).  
  - RobustScaler for outlier-resistant normalization.  
  - Sliding window sequence creation for time-series learning.  

- **Model Architecture**  
  - Bidirectional LSTM(128) → Bidirectional LSTM(64).  
  - Dense layers: 64 → 32 with dropout regularization.  
  - Adam optimizer, MSE loss, MAE metric.  

- **Training Enhancements**  
  - EarlyStopping with best weight restoration.  
  - ReduceLROnPlateau for dynamic learning rate tuning.  
  - Balanced epochs and batch size for efficiency.  

- **Evaluation & Visualization**  
  - Training vs validation loss curves.  
  - Actual vs Predicted PM2.5 scatter plots.  
  - Detailed progress report toward leaderboard target.  

---

## 📊 Results
- **Validation RMSE:** ~ *your_val_rmse_here*  
- **Expected Kaggle Public Score:** ~ *val_rmse × 54*  
- **Previous Best:** 3498
- **Target:** 2000  

Progress analysis confirms **significant improvement** toward the aggressive target.  

---

## 📁 Project Structure
```

├── data/
│   ├── train.csv
│   ├── test.csv
├── submissions/
│   ├── submission\_xxx.csv
│
├── README.md
└── requirements.txt

````

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/lscblack/Time-Series-Forecasting
cd pm25-bilstm
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run notebooks or scripts

* Start JupyterLab / Notebook and execute the provided notebooks.
* Adjust hyperparameters and features in `02_feature_engineering.ipynb` and `03_model_training.ipynb`.

---

## 📦 Requirements

* Python 3.10+
* TensorFlow / Keras
* NumPy, Pandas, Scikit-learn
* Matplotlib

Install all via:

```bash
pip install -r requirements.txt
```

---

## 📌 Next Steps

* Test longer sequence lengths (72–96 hours).
* Add attention layers for better temporal focus.
* Advanced hyperparameter tuning with grid/random search.
* Explore transformer-based architectures.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

Developed by **Chriss (Loue Sauveur Christian)**
Software Engineering @ African Leadership University

```
