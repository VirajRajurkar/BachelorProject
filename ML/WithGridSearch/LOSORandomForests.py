import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from a CSV file
data = pd.read_csv('/Users/viraj.rajurkar/Desktop/Bachelor Project/ManualFeatures/Feature_Matrix_Multiclass_Arabic_Split_1.csv')

# Display the first few rows of the dataset
print(data.head())

# Display the column names
print(data.columns)

# Specify feature columns and the target column
X = data.iloc[:, :-1].values  # All columns except the last one as features
y = data.iloc[:, -1].values   # Last column as the target variable

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Verify the encoding
print("Encoded labels:", np.unique(y))
print("Original labels:", label_encoder.classes_)

# Create the groups array
num_subjects = 5
rows_per_subject = 640
groups = np.repeat(np.arange(num_subjects), rows_per_subject)

# Check the shape of the dataset
print("Shape of feature matrix X:", X.shape)
print("Shape of label vector y:", y.shape)

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Initialize the classifier
clf = RandomForestClassifier(random_state=42)

# Set up the parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 200],           # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],         # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],         # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],           # Minimum samples required to be at a leaf node
    'bootstrap': [True, False]               # Whether bootstrap samples are used when building trees
}

# Use Leave-One-Subject-Out Cross-Validation
logo = LeaveOneGroupOut()

# Use GridSearchCV with LOSO Cross-Validation
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           cv=logo, scoring='accuracy', n_jobs=-1, verbose=2)

# Perform GridSearchCV on the entire dataset
grid_search.fit(X, y, groups=groups)

# Output the best parameters and best cross-validation score
print("\nBest Parameters found by Grid Search:", grid_search.best_params_)
print("Best Cross-Validation Accuracy from Grid Search:", grid_search.best_score_)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate each LOSO split
logo = LeaveOneGroupOut()
split_accuracies = []
confusion_matrices = []

for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    split_accuracies.append(accuracy)
    
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)
    
    print(f"Accuracy for current split: {accuracy}")
    print(f"Confusion Matrix for current split:\n{cm}")

# Print average accuracy across all splits
average_accuracy = np.mean(split_accuracies)
print(f"\nAverage LOSO Cross-Validation Accuracy: {average_accuracy}")

# Aggregate confusion matrices
total_cm = np.sum(confusion_matrices, axis=0)
print(f"\nAggregated Confusion Matrix:\n{total_cm}")