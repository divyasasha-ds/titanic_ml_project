# Step 1: Import required libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Upload the Titanic CSV file
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('train.csv')

# Step 3: Data understanding
print(df.head())
print(df.info())
print(df.isnull().sum())

# Step 4: Select features + handle missing values
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

data = df[features + [target]].copy()

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data_encoded = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
print(data_encoded.head())

# Step 5: Train-test split
X = data_encoded.drop(target, axis=1)
y = data_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6a: Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy:", acc_log)

# Step 6b: Decision Tree model
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print("Decision Tree Accuracy:", acc_tree)

# Step 7: Evaluation
print("\n=== Logistic Regression ===")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("\n=== Decision Tree ===")
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Step 8: Feature importance / coefficients
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': log_reg.coef_[0]
}).sort_values(by='coefficient', ascending=False)

print("\nLogistic Regression Coefficients:")
print(coef_df)

fi_df = pd.DataFrame({
    'feature': X.columns,
    'importance': tree_clf.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nDecision Tree Feature Importances:")
print(fi_df)

# Plot DT feature importance
plt.figure(figsize=(8, 4))
sns.barplot(data=fi_df, x='importance', y='feature')
plt.title("Decision Tree Feature Importances")
plt.tight_layout()
plt.show()
