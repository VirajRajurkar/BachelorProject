import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data from a CSV file
data = pd.read_csv('/Users/viraj.rajurkar/Desktop/SNN_ordered_after_subject_English.csv')

# Specify feature columns and the target column
X = data.iloc[:, :-1].values  # All columns except the last one as features
y = data.iloc[:, -1].values   # Last column as the target variable

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize k-NN classifier
knn = KNeighborsClassifier(metric = "precomputed")

# Use StratifiedKFold to ensure each fold is a representative sample
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform 5-fold cross-validation with predictions for each fold
y_pred = cross_val_predict(knn, X, y, cv=cv)

# Evaluate the classifier using the classification report
print("\nOverall Classification Report (Across All Folds):")
print(classification_report(y, y_pred))

# Confusion Matrix for all folds combined
print("\nOverall Confusion Matrix (Across All Folds):")
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)

# For each fold, print accuracy and confusion matrix
fold = 1
fold_accuracies = []
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train the model on the training data for the fold
    knn.fit(X_train, y_train)
    
    # Predict the test data for the fold
    fold_pred = knn.predict(X_test)
    
    # Print accuracy for this fold
    fold_accuracy = accuracy_score(y_test, fold_pred)
    print(f"\nFold {fold} Accuracy: {fold_accuracy:.2f}")
    fold_accuracies.append(fold_accuracy)

    # Print confusion matrix for this fold
    print(f"Fold {fold} Confusion Matrix:")
    cm = confusion_matrix(y_test, fold_pred)
    print(cm)

    fold += 1

# Calculate and print the mean accuracy across all folds
mean_accuracy = np.mean(fold_accuracies)
print(f"\nMean Cross-Validation Accuracy: {mean_accuracy:.2f}")