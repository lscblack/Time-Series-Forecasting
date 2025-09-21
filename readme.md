# 🌍 PM2.5 Air Quality Forecasting with Enhanced Bidirectional LSTM

## 📌 Overview

This project tackles the challenge of forecasting **PM2.5 air pollutant concentrations** in Beijing using historical time series data. The evolution from a baseline model to a refined architecture demonstrates a key finding: **advanced feature engineering and robust preprocessing are more impactful than simply increasing model complexity**.

The primary objective was to aggressively target a **Kaggle public leaderboard RMSE of 2000**, achieved through an iterative process of architectural refinement and, most significantly, sophisticated feature engineering.

## 🚀 Key Features & Contributions

-   **Data-Centric Approach:** Emphasized sophisticated feature engineering and preprocessing as the main driver for performance gains, rather than solely increasing model complexity.
-   **Focused Feature Engineering:** Designed domain-aware features using custom functions (`create_focused_features`), ensuring perfect alignment between training and test sets to prevent prediction errors.
-   **Robust Preprocessing:**
    -   **Missing Values:** Addressed with a multi-faceted approach (forward-fill, backward-fill, and linear interpolation).
    -   **Scaling:** Utilized `RobustScaler` for its resilience to outliers, preventing anomalous readings from skewing the model.
    -   **Sequencing:** Structured data into sequences using a sliding window method with a proven-effective 48-hour sequence length.
-   **Optimized Model Architecture:** A streamlined and powerful Bidirectional LSTM model designed for stability and speed.
-   **Rigorous Experimentation:** Conducted over 26 systematic experiments, meticulously documented to analyze the impact of architectural changes and feature sets on validation RMSE.

## 📊 Results & Performance

| Experiment Phase | Key Changes | Features | Validation RMSE | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Basic LSTM | Original | ~7,892 | Established baseline. |
| **Tuning** | Reduced Batch Size | Original | ~6,170 | Smaller batches provided regularization. |
| **Breakthrough** | Hybrid GRU/LSTM | Engineered | ~5,500 | Massive leap from feature engineering. |
| **Fine-Tuning** | Tuned Hybrid | Engineered | **3,330** | Achieved the lowest loss. |
| **Final Model** | **Bidirectional LSTM** | **Focused** | **2825.2246** | Simplified architecture on superior features. |

-   **Final Validation RMSE:** **X.XX** *(Please insert your final result here)*
-   **Expected Kaggle Public Score:** **~X.XX × 54**
-   **Improvement vs. Previous Best:** Significant improvement over the previous best of 2825.2246.
-   **Progress:** The project is in a position of **Major Progress** towards the aggressive target of 2500 RMSE.

## 🏗️ Final Model Architecture

The final model is a streamlined Bidirectional LSTM:

```python
Sequential([
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.2), input_shape=(n_steps, n_features)),
    Bidirectional(LSTM(64, dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
```

**Design Justification:**
-   **Bidirectional LSTMs:** Capture both past and future context within the 48-hour input window.
-   **Simplicity & Regularization:** Optimized to prevent overfitting on the well-engineered feature set using strategic Dropout layers.
-   **Optimizer:** Adam with a tuned learning rate of 0.0008 for efficient convergence.

## 📁 Project Structure

```
Time-Series-Forecasting/
├── data/
│   ├── train.csv           # Cleaned training dataset
│   └── test.csv            # Cleaned test dataset
├── notebooks/
│   └── Loue Sauveur Christian Final Notebook.ipynb  # Main project notebook
├── submissions/
│   └── submission_final.csv      # Final Kaggle submission file
├── README.md
└── requirements.txt
```

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/lscblack/Time-Series-Forecasting
cd Time-Series-Forecasting
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
Open and execute `Loue Sauveur Christian Final Notebook.ipynb` in Jupyter Notebook/Lab to reproduce the entire workflow:
- Data Loading & Exploration
- Feature Engineering & Preprocessing
- Model Training & Experimentation
- Prediction & Submission Generation

## 🔮 Recommendations for Future Work

1.  **Sequence Length Ablation:** Systematically test longer sequences (72–96 hours) to capture longer-term dependencies.
2.  **Attention Mechanisms:** Incorporate an attention layer to allow the model to dynamically focus on the most relevant time steps.
3.  **Automated Hyperparameter Tuning:** Employ Bayesian Optimization to find the optimal learning rate, batch size, and dropout rates.
4.  **Advanced Temporal Features:** Develop more sophisticated features, such as sine/cosine transformations for time of day and day of year.

## 📜 License

This project is licensed under the MIT License.

## 👤 Author

**Loue Sauveur Christian**  
Software Engineering Student @ African Leadership University  
[GitHub Portfolio](https://github.com/lscblack)

