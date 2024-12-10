import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/viraj.rajurkar/Desktop/Bachelor Project/CNNFeatures/CNN_Feature_Matrix_Arabic.csv'
data = pd.read_csv(file_path)

print(data)

# Convert string labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['labels'])  # Convert 'NEng', 'YEng' to integers

# Separate features (exclude the labels column)
X = data.iloc[:, :-1].values  # All columns except the last one

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the cross-validation strategy (10-Fold Stratified)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define SVM and the hyperparameter grid
param_grid = {
    'C': [0.01,0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
grid.fit(X_scaled, y)

# Output the best hyperparameters and cross-validation accuracy
print("Best Parameters from GridSearchCV:", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_)

# Use the best model from GridSearchCV
best_model = grid.best_estimator_

# Perform 5-Fold Cross-Validation and display accuracy for each fold
fold_accuracies = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y), start=1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    best_model.fit(X_train, y_train)  # Train on the current fold
    y_fold_pred = best_model.predict(X_test)  # Predict on the current fold
    fold_accuracy = accuracy_score(y_test, y_fold_pred)
    fold_accuracies.append(fold_accuracy)
    print(f"Accuracy for Fold {fold}: {fold_accuracy:.2f}")

# Display the mean accuracy across all folds
mean_accuracy = np.mean(fold_accuracies)
print(f"\nMean Cross-Validation Accuracy: {mean_accuracy:.2f}")

# Evaluate using cross-validated predictions for confusion matrix and classification report
y_pred = cross_val_predict(best_model, X_scaled, y, cv=cv, n_jobs=-1)

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
print(f"\nCross-Validation Accuracy (Using Cross-Val Predict): {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()
