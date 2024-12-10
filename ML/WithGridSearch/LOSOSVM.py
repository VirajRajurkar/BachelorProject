import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from a CSV file
data = pd.read_csv('/Users/viraj.rajurkar/Desktop/Bachelor Project/CNNFeatures/CNN_ordered_after_subject_Arabic.csv')
#data = data.sample(frac=1).reset_index(drop=True)

# Display the first few rows of the dataset
print(data.head())

# Display the column names
print(data.columns)

# Specify feature columns and the target column
X = data.iloc[:, :-1].values  # All columns except the last one as features
y = data.iloc[:, -1].values   # Last column as the target variable

# Encode the labels with NEng as 1 and YEng as 0
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

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the SVM classifier
svm_clf = SVC(class_weight='balanced')

# Set up the parameter grid for grid search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'poly', 'rbf']
}

# Use Leave-One-Subject-Out Cross-Validation
logo = LeaveOneGroupOut()

# Use GridSearchCV with LOSO Cross-Validation
grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid, 
                           cv=logo, scoring='accuracy', n_jobs=-1, verbose=2)

# Perform GridSearchCV on the entire dataset
grid_search.fit(X_scaled, y, groups=groups)

# Output the best parameters and best cross-validation score
print("\nBest Parameters found by Grid Search:", grid_search.best_params_)
print("Best Cross-Validation Accuracy from Grid Search:", grid_search.best_score_)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate each LOSO split
split_accuracies = []
confusion_matrices = []

for train_index, test_index in logo.split(X_scaled, y, groups):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
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