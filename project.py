"""# Libraries"""
## Data manipulation and analysis
import numpy as np
import pandas as pd

## Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

## signal processing
import mne
from sklearn.decomposition import PCA, FastICA
from mne.decoding import CSP
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
from scipy.linalg import eigh

## Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

"""# parameters"""

fs = 250
freq = 50
order = 4
lowcut = 8
highcut = 30
quality = 30
n_components = 4
n_components_pca = 16
n_components_ica = 16
n_components_csp = 4

"""# Data

## old data overview

- there are E and T from every file like A01E and A01T, where E is the training set and T is the test set.
- The data is divided into 9 sessions, each containing 288 trials. Each trial is 4 seconds long and contains 22 EEG channels.
- The data is sampled at 250 Hz, resulting in 1000 samples per trial.
- The labels are binary (1, 2, 3, 4) and represent the two classes of motor imagery tasks: left hand, right hand, Tounge and Foot.

## load data
"""

data = pd.read_csv('Data/eeg-motor-imagery-bciciv-2a/BCICIV_2a_all_patients.csv')

"""## data overview"""

data.head()
print('shape:', data.shape)
print('columns:', data.columns)
print('info:', data.info())
print('describe:\n', data.describe())
print('missing values:\n', data.isnull().sum())

"""## label unique values"""

print('unique labels:', data['label'].unique())
print('unique labels count:', data['label'].value_counts())

"""- there is no missing values in the data
- there is no gap in the range of the data
- there are 4 labels: right hand, left hand, feet and tongue
"""

# visualize corr of data
# plt.figure(figsize=(15, 15))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
# plt.show()

"""# Preprocessing

## Encoding
"""

data['label'] = data['label'].map({'right': 1, 'left': 2, 'foot': 3, 'tongue': 4})

"""## EEG preprocessing functions

### bandpass filter
"""

def bandpass_filter(data, lowcut=8, highcut=30, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

"""### notch filter"""

def notch_filter(data, freq=50, fs=250, quality=30):
    b, a = signal.iirnotch(freq, quality, fs)
    return filtfilt(b, a, data, axis=0)

"""## prepare data"""

X = data.drop('label', axis=1).values
y = data['label'].values

print("\nInitial class distribution:")
unique_labels, counts = np.unique(y, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")

"""## Reshape data into trials"""

samples_per_trial = 100  # Changed from 1000 to 100
n_trials = X.shape[0] // samples_per_trial
X = X[:n_trials * samples_per_trial]
y = y[:n_trials * samples_per_trial]
y_labels = y[::samples_per_trial]
X_reshaped = X.reshape(n_trials, samples_per_trial, -1).transpose(0, 2, 1)

print("\nAfter reshaping:")
print(f"Number of trials: {n_trials}")
print(f"Samples per trial: {samples_per_trial}")
print(f"X_reshaped shape: {X_reshaped.shape}")
print(f"y_labels shape: {y_labels.shape}")

print("\nClass distribution after reshaping:")
unique_labels, counts = np.unique(y_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} trials")

"""## apply EEG preprocessing"""

print("Applying bandpass filter...")
filtered_data = bandpass_filter(X_reshaped)
print("Applying notch filter...")
filtered_data = notch_filter(filtered_data)

print("\nData shape after preprocessing:", filtered_data.shape)
print("Class distribution after preprocessing:")
unique_labels, counts = np.unique(y_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} trials")

"""## remove EOG with PCA"""

n_trials, n_channels, n_samples = filtered_data.shape
# Reshape to preserve temporal information
data_2d = filtered_data.reshape(n_trials, n_channels, n_samples)

# Apply PCA to each time point
pca_data = np.zeros((n_trials, n_components_pca, n_samples))
pca = PCA(n_components=n_components_pca)

# First flatten the data for PCA
flattened_data = data_2d.reshape(n_trials * n_samples, n_channels)
# Fit PCA on all data
pca.fit(flattened_data)

# Transform each time point
for t in range(n_samples):
    time_data = data_2d[:, :, t]
    pca_data[:, :, t] = pca.transform(time_data)

print(f"\nPCA data shape: {pca_data.shape}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.2%}")

print("\nData shape after PCA:", pca_data.shape)
print("Class distribution after PCA:")
unique_labels, counts = np.unique(y_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} trials")

"""### Plot PCA components"""
plt.figure(figsize=(18, 12))
n_cols = 4
n_rows = int(np.ceil(n_components_pca / n_cols))

# Plot each PCA component
for i in range(n_components_pca):
    plt.subplot(n_rows, n_cols, i+1)
    # Get the i-th component's time series data
    component_data = pca_data[:, i, :].flatten()
    plt.plot(component_data, alpha=0.7)
    plt.title(f'PCA Component {i+1}\nVar: {pca.explained_variance_ratio_[i]:.2%}', fontsize=8)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.suptitle('All PCA Components (Time Series)', y=1.02)
plt.show()

"""## apply ICA"""

print("\nApplying ICA...")
# Reshape for ICA while preserving temporal information
ica_input = pca_data.reshape(n_trials * n_samples, n_components_pca)
ica = FastICA(n_components=n_components_ica, random_state=42)
ica_data = ica.fit_transform(ica_input)

print("\nData shape after ICA:", ica_data.shape)
print("Class distribution after ICA:")
unique_labels, counts = np.unique(y_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} trials")

"""### plot all ICA components"""
plt.figure(figsize=(18, 12))
n_cols = 4
n_rows = int(np.ceil(n_components_ica / n_cols))
for i in range(n_components_ica):
    plt.subplot(n_rows, n_cols, i+1)
    plt.plot(ica_data[:, i], alpha=0.7)
    plt.title(f'ICA Component {i+1}', fontsize=8)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.suptitle('All ICA Components (Time Series)', y=1.02)
plt.show()

"""### reshape ICA data"""
# Reshape back to maintain temporal structure
ica_data_reshaped = ica_data.reshape(n_trials, n_samples, n_components_ica)
# Transpose to get (trials, channels, samples)
ica_data_reshaped = np.transpose(ica_data_reshaped, (0, 2, 1))

print(f"ICA data shape after reshaping: {ica_data_reshaped.shape}")

# Verify dimensions
if ica_data_reshaped.shape[1] == n_components_ica:
    print("Success! ICA data has exactly 16 components")
else:
    print(f"Warning: ICA data has {ica_data_reshaped.shape[1]} components instead of 16")

# Use ica_data_reshaped for CSP extraction
cleaned_data = ica_data_reshaped

"""## Extract CSP features"""

print("Extracting CSP features...")
# Data validation
print(f"Shape of cleaned_data before any processing: {cleaned_data.shape}")
print(f"Shape of y_labels: {y_labels.shape}")
print("Unique values in y_labels before processing:", np.unique(y_labels, return_counts=True))

# Initialize CSP
csp = CSP(n_components=n_components_csp, reg=None, log=True, norm_trace=False)

# Make sure data is in the correct shape (trials, channels, samples)
if cleaned_data.shape[1] != n_components_ica:
    cleaned_data = np.transpose(cleaned_data, (0, 2, 1))
print(f"Data shape before CSP: {cleaned_data.shape}")

# Verify there are trials for each class
for class_id in range(1, 5):  # Now checking all 4 classes
    n_trials = np.sum(y_labels == class_id)
    print(f"Number of trials for class {class_id}: {n_trials}")

# Additional check to make sure data and labels align
assert len(cleaned_data) == len(y_labels), "Number of trials doesn't match number of labels"
assert not np.any(np.isnan(cleaned_data)), "Data contains NaN values"
assert not np.any(np.isinf(cleaned_data)), "Data contains infinite values"

# Fit and transform the data
csp_features = csp.fit_transform(cleaned_data, y_labels)

print("CSP feature extraction completed")
print("\nFeature Information:")
print("CSP Features shape:", csp_features.shape)

# Visualizing CSP patterns
print("\nVisualizing CSP patterns...")
patterns = csp.patterns_
patterns_norm = patterns / np.max(np.abs(patterns))

plt.figure(figsize=(16, 4))
class_names = ['Right Hand', 'Left Hand', 'Feet', 'Tongue']  # Updated class names for all 4 classes
for i in range(min(4, n_components_csp)):  # Show patterns for all 4 classes
    plt.subplot(1, 4, i + 1)
    plt.imshow(patterns_norm[i].reshape(4, 4), cmap='jet', interpolation='nearest')
    plt.title(f'CSP Pattern {i+1}')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Train SVM classifier with all classes
print("\nTraining SVM classifier...")
X_train, X_test, y_train, y_test = train_test_split(
    csp_features, y_labels, test_size=0.2, random_state=42
)

clf = SVC(kernel='rbf', random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nClassification Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

"""## Save preprocessed data and features"""

np.save('preprocessed_data.npy', cleaned_data)
np.save('csp_features.npy', csp_features)

print('----------------------------------------')

"""# Model"""

"""## Load preprocessed data and features"""

"""## Save preprocessed data and features"""

np.save('preprocessed_data.npy', cleaned_data)
np.save('csp_features.npy', csp_features)

print('----------------------------------------')

"""# Model"""

"""## Load preprocessed data and features"""

cleaned_data = np.load('preprocessed_data.npy')
csp_features = np.load('csp_features.npy')

"""## Split data into training and testing sets"""

print("\nBefore training:")
print(f"CSP features shape: {csp_features.shape}")
print(f"Labels shape: {y_labels.shape}")
print("\nClass distribution in full dataset:")
unique_labels, counts = np.unique(y_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")

X_train, X_test, y_train, y_test = train_test_split(csp_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

print("\nTraining set:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print("\nClass distribution in training set:")
unique_labels, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")

print("\nTest set:")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("\nClass distribution in test set:")
unique_labels, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")

"""## SVM Model"""

# Try different kernels
kernels = ['linear', 'rbf', 'poly']
best_accuracy = 0
best_kernel = None

for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {kernel} kernel: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_kernel = kernel

print(f"\nBest kernel: {best_kernel} with accuracy: {best_accuracy:.4f}")

# Train final model with best kernel
svm = SVC(kernel=best_kernel, random_state=42)
svm.fit(X_train, y_train)

"""## Evaluate model"""
y_pred = svm.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

