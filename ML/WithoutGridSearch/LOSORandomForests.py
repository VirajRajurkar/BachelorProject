import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from a CSV file
data = pd.read_csv('/Users/viraj.rajurkar/Desktop/Bachelor Project/ManualFeatures/Feature_Matrix_Multiclass_Arabic_Split_1.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display the column names
print("\nColumn names:")
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

# Create the groups array for Leave-One-Subject-Out
num_subjects = 5
rows_per_subject = 640
groups = np.repeat(np.arange(num_subjects), rows_per_subject)

# Check the shape of the dataset
print("Shape of feature matrix X:", X.shape)
print("Shape of label vector y:", y.shape)

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Initialize the Random Forest classifier with the specified parameters
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    bootstrap=False,
    random_state=42
)

# Use Leave-One-Subject-Out Cross-Validation
logo = LeaveOneGroupOut()

# Evaluate each LOSO split
split_accuracies = []
confusion_matrices = []

for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    
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