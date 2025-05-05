# 💧 Water Quality Prediction Model

This project focuses on predicting water potability using machine learning techniques. By analyzing various physicochemical properties of water, the model determines whether the water is safe for consumption.

# 📊 Dataset

The dataset used is `water_potability.csv`, which contains features such as:

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity
- Potability (target variable)
The target variable `Potability` indicates whether the water is safe to drink (1) or not (0).

# 🧪 Project Structure

<pre>Water-Prediction-Model/
├── anova_rfe.py                     # Feature selection using ANOVA and RFE
├── water_quality_logistic.py        # Logistic Regression implementation
├── Water_potability_RandomForest.py # Random Forest implementation
├── README.md                        # Project documentation</pre>

# ⚙️ Features

- Data preprocessing and handling missing values
- Feature selection using ANOVA and Recursive Feature Elimination (RFE)
- Model training using Logistic Regression and Random Forest
- Evaluation metrics: Accuracy, Confusion Matrix, Classification Report
- Visualization of results

# 🚀 Getting Started

**Prerequisites**
- Python 3.x
- Required libraries: pandas, numpy, scikit-learn, matplotlib

**Installation**

1. Clone the repository:
<pre> git clone https://github.com/WASD-su65/Water-Prediction-Model.git </pre>

2. Navigate to the project directory:
<pre> cd Water-Prediction-Model </pre>

3. Install the required libraries:
<pre> pip install -r requirements.txt </pre>

4. Run the desired Python script:
<pre> python anova_rfe.py </pre>

or

<pre> python water_quality_logistic.py </pre>

or

<pre> python Water_potability_RandomForest.py </pre>

# 📈 Results

The models are evaluated using accuracy scores, confusion matrices, and classification reports to determine their performance in predicting water potability.

# 📄 License

No specific license has been provided for this project. Please contact the project owner before using or redistributing.
