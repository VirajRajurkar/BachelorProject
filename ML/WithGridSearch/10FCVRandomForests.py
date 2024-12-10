import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from a CSV file
data = pd.read_csv('/Users/viraj.rajurkar/Desktop/Bachelor Project/ManualFeatures/Feature_Matrix_Multiclass_Japanese_Split_5.csv')

# Specify feature columns and the target column
X = data.iloc[:, :-1].values  # All columns except the last one as features
y = data.iloc[:, -1].values   # Last column as the target variable

# Check the shape of the dataset
print("Shape of feature matrix X:", X.shape)
print("Shape of label vector y:", y.shape)

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Define the cross-validation strategy (10-Fold Stratified)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, 
                           cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV on the entire dataset
grid_search.fit(X, y)

# Output the best parameters and best cross-validation score
print("\nBest Parameters found by Grid Search:", grid_search.best_params_)
print("Best Cross-Validation Accuracy from Grid Search:", grid_search.best_score_)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Perform 10-Fold Cross-Validation and display accuracy for each fold
split_accuracies = []
confusion_matrices = []

for fold, (train_index, test_index) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
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
y_pred_all = cross_val_predict(best_model, X, y, cv=cv)
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