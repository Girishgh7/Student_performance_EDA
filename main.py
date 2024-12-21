# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import randint

# Load the dataset
file_path = "/kaggle/input/students-performance-10000-clean-data-eda/students_performance.csv"  # Update this path
data = pd.read_csv(file_path)

# Preprocess the data
# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
target_column = "target_column"  # Replace with your actual target column
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a basic Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the basic model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Basic Decision Tree Accuracy: {accuracy:.2f}")

# Hyperparameter Tuning using Grid Search
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Parameters from Grid Search:", grid_search.best_params_)
print("Best Score from Grid Search:", grid_search.best_score_)

# Hyperparameter Tuning using Randomized Search
param_dist = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'criterion': ['gini', 'entropy']
}
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)
print("Best Parameters from Randomized Search:", random_search.best_params_)
print("Best Score from Randomized Search:", random_search.best_score_)

# Feature Importances
clf_best = grid_search.best_estimator_
importances = clf_best.feature_importances_
feature_names = X.columns

print("\nFeature Importances:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Feature Importances in Decision Tree")
plt.show()

# Visualize the decision tree
class_names = [str(cls) for cls in clf_best.classes_]
plt.figure(figsize=(20, 10))
plot_tree(clf_best, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()
