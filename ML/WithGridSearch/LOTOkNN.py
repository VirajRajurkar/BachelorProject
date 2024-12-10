import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from a CSV file
data = pd.read_csv('')

# Specify feature columns and the target column
X = data.iloc[:, :-1].values  # All columns except the last one as features
y = data.iloc[:, -1].values   # Last column as the target variable

# Check the shape of the dataset
print("Shape of feature matrix X:", X.shape)
print("Shape of label vector y:", y.shape)

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Define the cross-validation strategy (5-Fold Stratified)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the Random Forest classifier with chosen parameters
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=False,
    random_state=42
)

# Perform 5-Fold Cross-Validation and display accuracy for each fold
split_accuracies = []
confusion_matrices = []

for fold, (train_index, test_index) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    split_accuracies.append(accuracy)
    
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)
    
    print(f"Accuracy for fold {fold}: {accuracy:.2f}")
    print(f"Confusion Matrix for fold {fold}:\n{cm}")

# Print average accuracy across all splits
average_accuracy = np.mean(split_accuracies)
print(f"\nAverage 5-Fold Cross-Validation Accuracy: {average_accuracy:.2f}")

# Aggregate confusion matrices
total_cm = np.sum(confusion_matrices, axis=0)
print(f"\nAggregated Confusion Matrix:\n{total_cm}")

# Detailed classification report
y_pred_all = cross_val_predict(rf_clf, X, y, cv=cv)
print("\nClassification Report:")
print(classification_report(y, y_pred_all))

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred_all)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix with Labels (Best Model)')
plt.show()