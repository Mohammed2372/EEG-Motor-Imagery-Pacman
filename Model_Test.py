"""# Test Script for EEG Motor Imagery Classification"""
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, classification_report


print("Loading test data...")

X_test = np.load('Saved Data/X_test.npy')
y_test = np.load('Saved Data/y_test.npy')

print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)
print("Test data loaded successfully.")

print("\nLoading model...")
model = load('svm_model.joblib')

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(X_test)

# Save predictions
## Map predictions back to string labels before saving
label_map = {1: 'right', 2: 'left', 3: 'down', 4: 'up'}
predictions_str = np.vectorize(label_map.get)(predictions)

np.save('Saved Data/predictions.npy', predictions_str)
print("Predictions saved to 'predictions.npy' as string labels")

# Evaluate results
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Detailed classification report
class_names = ['Right Hand', 'Left Hand', 'Feet', 'Tongue']
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=class_names))
