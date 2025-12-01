# Titanic Survival Prediction

This project predicts whether a passenger survived the Titanic disaster using Machine Learning.

## Dataset
The dataset is taken from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data).

### Features used:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

### Target:
- Survived (1 = survived, 0 = died)

## Data Preprocessing
- Handled missing values for Age (median) and Embarked (mode)
- Converted categorical variables (Sex, Embarked) to numeric using one-hot encoding

## Models Used
1. Logistic Regression
2. Decision Tree Classifier

## Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance

## Libraries Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## How to Run
1. Upload `train.csv` to Colab
2. Open `titanic_survival_prediction.ipynb`
3. Run all cells to see predictions and feature importance
