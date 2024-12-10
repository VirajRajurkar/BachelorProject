import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Set random state for reproducibility
random_state = 42

# Load data from a CSV file
data = pd.read_csv('/Users/viraj.rajurkar/Desktop/Bachelor Project/CNNFeatures/CNN_Feature_Matrix_Arabic.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display the column names
print("\nColumn names:")
print(data.columns)

# Display basic information about the dataset
print("\nBasic information about the dataset:")
print(data.info())

# Specify feature columns and the target column
X = data.iloc[:, :-1].values  # All columns except the last one as features
y = data.iloc[:, -1].values   # Last column as the target variable

# Encode the labels  as 1 and  as 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Verify the encoding
print("Encoded labels:", np.unique(y))
print("Original labels:", label_encoder.classes_)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the cross-validation strategy (10-Fold Stratified)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

# Initialize the SVM classifier with the best parameters found by GridSearch
svm_clf = SVC(C=10, gamma=0.001, kernel='rbf', class_weight='balanced', random_state=random_state)

# Perform 10-Fold Cross-Validation and display accuracy for each fold
split_accuracies = []
confusion_matrices = []

for train_index, test_index in cv.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    split_accuracies.append(accuracy)
    
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)
    
    print(f"Accuracy for current split: {accuracy}")
    print(f"Confusion Matrix for current split:\n{cm}")

# Print average accuracy across all splits
average_accuracy = np.mean(split_accuracies)
print(f"\nAverage 10-Fold Cross-Validation Accuracy: {average_accuracy}")

# Aggregate confusion matrices
total_cm = np.sum(confusion_matrices, axis=0)
print(f"\nAggregated Confusion Matrix:\n{total_cm}")


# Plot the aggregated confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Aggregated Confusion Matrix')
plt.show()