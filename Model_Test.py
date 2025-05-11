"""# Test Script for EEG Motor Imagery Classification"""
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, classification_report


print("Loading test data...")

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)
print("Test data loaded successfully.")

print("\nLoading model...")
model = load('svm_model.joblib')

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(X_test)

# Save predictions
np.save('predictions.npy', predictions)
print("Predictions saved to 'predictions.npy'")

# Evaluate results
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Detailed classification report
class_names = ['Right Hand', 'Left Hand', 'Feet', 'Tongue']
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=class_names))

# Save results to a text file
with open('test_results.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, predictions, target_names=class_names))

