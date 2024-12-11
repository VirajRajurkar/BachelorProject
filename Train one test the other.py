# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:13:05 2024
@author: SKV Hähnlein
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from separate CSV files for training (first language) and testing (second language)
train_data = pd.read_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_CNN/CNN_Binary/CNN Unordered/CNN_Feature_Matrix_Japanese.csv')
test_data = pd.read_csv('C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_CNN/CNN_Binary/CNN Ordered after trial/CNN_ordered_after_Trial_English.csv')


# Specify feature columns and the target column for training data (first language)
X_train = train_data.iloc[:, :-1].values  # All columns except the last one as features
y_train = train_data.iloc[:, -1].values   # Last column as the target variable

# Specify feature columns and the target column for testing data (second language)
X_test = test_data.iloc[:, :-1].values    # All columns except the last one as features
y_test = test_data.iloc[:, -1].values     # Last column as the target variable

# Function to encode labels (1 for 'N' and 0 for others)
def encode_labels(labels):
    return np.array([1 if str(label).startswith('N') else 0 for label in labels])

# Apply encoding to the training and testing labels
y_train = encode_labels(y_train)
y_test = encode_labels(y_test)

# Create the groups array for Leave-One-Subject-Out
num_subjects = 10
rows_per_subject = 320
groups = np.repeat(np.arange(num_subjects), rows_per_subject)

# Check class distribution
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

print(f"Class distribution in training data: {dict(zip(unique_train, counts_train))}")
print(f"Class distribution in testing data: {dict(zip(unique_test, counts_test))}")



# Initialize LeaveOneGroupOut for LOSO cross-validation
logo = LeaveOneGroupOut()

# Initialize models for evaluation
svm_clf = SVC(C=10, gamma=0.001, kernel='rbf', class_weight='balanced', probability=True, random_state=42)


# Initialize lists to store metrics
split_accuracies = []
roc_aucs = []
confusion_matrices = []

# Perform Leave-One-Subject-Out Cross-Validation on the second language data (test set)
for train_index, test_index in logo.split(X_test, y_test, groups=groups):
    X_train_fold, X_test_fold = X_train, X_test[test_index]  # Use the whole first language data for training
    y_train_fold, y_test_fold = y_train, y_test[test_index]  # Use the corresponding test labels
    
    # Standardize the features (important for SVM and KNN)
    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)  # Standardize on the whole training data (first language)
    X_test_fold = scaler.transform(X_test_fold)  # Standardize test fold (second language subject)

    # Train the model on the first language data
    svm_clf.fit(X_train_fold, y_train_fold)
    
    # Predict and evaluate
    y_pred = svm_clf.predict(X_test_fold)
    y_pred_probs = svm_clf.predict_proba(X_test_fold)[:, 1]  # Probabilities for the positive class
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_fold, y_pred)
    split_accuracies.append(accuracy)
    
    # Calculate AUC-ROC score
    auc_score = roc_auc_score(y_test_fold, y_pred_probs)
    roc_aucs.append(auc_score)
    
    # Confusion matrix for the current fold
    cm = confusion_matrix(y_test_fold, y_pred)
    confusion_matrices.append(cm)
    
    

# Calculate overall metrics
print("\nOverall Cross-Validation Results:")
print(f"Median Accuracy: {np.median(split_accuracies)}")
print(f"Median AUC-ROC: {np.median(roc_aucs)}")
print(f"Confusion Matrices:\n{np.mean(confusion_matrices, axis=0)}")



# Set the style to "whitegrid" for a cleaner look
sns.set(style="whitegrid")

# Create a larger figure for better readability
plt.figure(figsize=(8, 6))

# Boxplot for accuracy distribution
sns.boxplot(data=split_accuracies, palette="Set2", 
            medianprops=dict(color='blue', linewidth=3),  # Highlight median
            flierprops=dict(marker='o', color='red', markersize=8))

# Calculate the median value
median_value = np.median(split_accuracies)

plt.title('Distribution of Accuracy across Trials Japanese vs English', fontsize=16)
plt.xlabel('Trials', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.tight_layout()
plt.legend([f'Median: {median_value:.2f}'], loc='upper left', fontsize=12)
plt.show()



# Boxplot for AUC-ROC distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data=roc_aucs, palette="Set2", 
            medianprops=dict(color='blue', linewidth=3),  # Highlight median
            flierprops=dict(marker='o', color='red', markersize=8))

# Calculate the median value
median_value = np.median(roc_aucs)

plt.title('Distribution of AUC-ROC across Trials Japanese vs English', fontsize=16)
plt.xlabel('Trials', fontsize=14)
plt.ylabel('AUC-ROC', fontsize=14)
plt.tight_layout()
plt.legend([f'Median: {median_value:.2f}'], loc='upper left', fontsize=12)
plt.show()
