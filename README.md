# 🏥 Medical Insurance Cost Prediction using Linear Regression

This project predicts individual medical insurance charges based on demographic and health-related features such as age, gender, BMI, number of children, smoking status, and region. It uses **Linear Regression** as the core algorithm and includes data visualization and preprocessing steps.

---

## 📁 Dataset

- **Filename**: `Medical_insurance.csv`
- **Features**:
  - `age`: Age of the individual
  - `sex`: Gender (`male` / `female`)
  - `bmi`: Body Mass Index
  - `children`: Number of children covered by insurance
  - `smoker`: Smoking status (`yes` / `no`)
  - `region`: Residential area (`southeast`, `southwest`, `northeast`, `northwest`)
  - `charges`: Medical insurance cost (target variable)

---

## 🚀 How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/medical-insurance-predictor.git
   cd medical-insurance-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script**
   ```bash
   python main.py
   ```

> ⚠️ Make sure `Medical_insurance.csv` is in the same directory as your script.

---

## 🛠️ Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 📊 Exploratory Data Analysis

- Age and BMI distributions (using `sns.displot`)
- Gender distribution (`sns.countplot`)
- Summary statistics (`dataframe.describe()`)

---

## 🧠 Model

- **Algorithm**: Linear Regression
- **Target Variable**: `charges`
- **Features Used**: All columns except `charges` (encoded where needed)
- **Performance Metric**: R² score

---

## 🧪 Sample Prediction

```python
input_data = (19, 0, 24.6, 1, 0, 0)  # Example input
prediction = model.predict([input_data])
print("Predicted Charges:", prediction)
```

---

## ✅ Results

- R² Score on Training Data: ~0.75+
- R² Score on Testing Data: ~0.70+

