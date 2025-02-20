# 📊 Hyperparameter Optimization for Supervised Learning Models

## 🧠 Project Overview
This project focuses on optimizing hyperparameters in supervised learning models, particularly **XGBoost**, using **Operations Research (OR) techniques**. The goal is to **minimize the Mean Squared Error (MSE)** on the **California Housing Dataset** while comparing the efficiency of various optimization strategies.

The main methods used in this project include:
- ✅ **Gaussian Process Regression (GPR)** as a surrogate model
- ✅ **L-BFGS-B Optimization** for fine-tuning
- ✅ **Bayesian Optimization**, **Grid Search**, and **Random Search** for benchmarking

The project highlights how **OR methods** can enhance hyperparameter optimization beyond traditional techniques.

📖 **For more details, read the full report in** [`docs/report_v_fr.pdf`](docs/report_v_fr.pdf).

---

## 📂 Table of Contents
- [📊 Project Overview](#-project-overview)
- [📁 Dataset](#-dataset)
- [⚙️ Methods Used](#️-methods-used)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [💻 How to Run](#-how-to-run)
- [📂 Results](#-results)
- [🔬 Key Insights](#-key-insights)
- [💡 Future Work](#-future-work)
- [📜 License](#-license)

---

## 📁 Dataset
The project uses the **California Housing Dataset** from `sklearn.datasets`:
```python
from sklearn.datasets import fetch_california_housing
```
- **Features**: 8 numeric features related to housing (e.g., median income, house age).
- **Target**: Median house value in California districts.
- **Total Samples**: ~20,000
- No external dataset download is required; it is fetched directly using scikit-learn.

---

## ⚙️ Methods Used

### 🏆 Hyperparameter Optimization Techniques
- **Gaussian Process Regression (GPR)**
  - Used as a surrogate model to approximate the objective function.
  - Helps in exploring the hyperparameter space more effectively.
- **L-BFGS-B Optimization**
  - A bounded optimization algorithm used for fine-tuning hyperparameters after GPR.
  - Efficient for large-scale problems with box constraints.
- **Baseline Methods for Comparison**
  - Grid Search
  - Random Search
  - Bayesian Optimization (`skopt.BayesSearchCV`)

### 📋 Hyperparameters Optimized
- `n_estimators` (50–300)
- `max_depth` (3–10)
- `min_child_weight` (1–5)
- `learning_rate` (0.01–0.3)
- `subsample` (0.6–1.0)

---

## 📈 Evaluation Metrics
- **Mean Squared Error (MSE)**: The primary metric used for evaluation.
- **Computation Time**: Measured to compare the efficiency of different optimization techniques.

---

## 💻 How to Run

### ✅ 1️⃣ Clone the Repository
```bash
git clone https://github.com/Salwa08/OR-Hyperparameter-Optimization.git
cd OR-Hyperparameter-Optimization
```

### ✅ 2️⃣ Set Up the Environment
It’s recommended to use a virtual environment:
```bash
python -m venv venv
# Activate the environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### ✅ 3️⃣ Run the Jupyter Notebook
```bash
jupyter notebook notebooks/hyperparameter_optimization.ipynb
```
Run the cells in order to:
- Load the dataset
- Initialize the optimizer
- Perform hyperparameter tuning
- Generate visualizations

---

## 📂 Results
- **Best MSE Achieved**: 0.1899
- **Optimized Hyperparameters**: Stored in `results/optimization_results.json`
- **Visualizations Generated**:
  - MSE Evolution over iterations
  - Hyperparameter Distributions
  - Correlations between hyperparameters and MSE

### Sample JSON Output:
```json
{
    "OR_Optimization": {
        "MSE": 0.18999189381099146,
        "Time": 147.54119777679443,
        "Parameters": {
            "n_estimators": 192,
            "max_depth": 5,
            "min_child_weight": 1,
            "learning_rate": 0.10483060403018869,
            "subsample": 0.7406229768820679
        }
    }
}
```

---

## 🔬 Key Insights
- Gaussian Process + L-BFGS-B outperformed basic Grid Search in both efficiency and MSE reduction.
- The Bayesian Optimization method closely followed the OR-based approach but required more iterations to converge.
- Certain hyperparameters (`learning_rate` and `subsample`) had a stronger impact on the MSE than others.

---

## 💡 Future Work
- 🔄 Apply this OR-based optimization technique to classification tasks.
- 📊 Integrate more complex surrogate models (e.g., Random Forests, Neural Networks).
- ⚡ Explore distributed optimization methods for larger datasets.

---

## 📜 License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
