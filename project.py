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

data['label'] = data['label'].map({'right': 1, 'left': 2, 'feet': 3, 'tongue': 4})

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

"""## Reshape data into trials"""

samples_per_trial = 1000
n_trials = X.shape[0] // samples_per_trial
X = X[:n_trials * samples_per_trial]
y = y[:n_trials * samples_per_trial]
y_labels = y[::samples_per_trial]
X_reshaped = X.reshape(n_trials, samples_per_trial, -1).transpose(0, 2, 1)

"""## apply EEG preprocessing"""

print("Applying bandpass filter...")
filtered_data = bandpass_filter(X_reshaped)
print("Applying notch filter...")
filtered_data = notch_filter(filtered_data)

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

"""## apply ICA"""

print("\nApplying ICA...")
# Reshape for ICA while preserving temporal information
ica_input = pca_data.reshape(n_trials * n_samples, n_components_pca)
ica = FastICA(n_components=n_components_ica, random_state=42)
ica_data = ica.fit_transform(ica_input)

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
    print("✅ Success! ICA data has exactly 16 components")
else:
    print(f"❌ Warning: ICA data has {ica_data_reshaped.shape[1]} components instead of 16")

# Use ica_data_reshaped for CSP extraction
cleaned_data = ica_data_reshaped

"""## Extract CSP features"""

print("Extracting CSP features...")
# Ensure data is in the correct format (trials, channels, samples)
if cleaned_data.shape[1] != n_channels:
    cleaned_data = np.transpose(cleaned_data, (0, 2, 1))
print(f"Data shape before CSP: {cleaned_data.shape}")

"""### Reshape data for CSP"""

n_trials, n_channels, n_samples = cleaned_data.shape
data_2d = cleaned_data.reshape(n_trials, n_channels * n_samples)
print(f"Data shape after reshaping: {data_2d.shape}")

"""### Calculate covariance matrices for each class"""
classes = np.unique(y_labels)  # Use y_labels instead of y
batch_size = 100  # Process data in batches to save memory

"""### Initialize arrays for storing class-specific data"""
class_covs = []
class_counts = []

for c in classes:
    # Get data for this class using y_labels
    class_data = data_2d[y_labels == c]
    n_class_trials = class_data.shape[0]

    # Initialize covariance matrix
    cov = np.zeros((n_channels * n_samples, n_channels * n_samples))

    # Process in batches
    for i in range(0, n_class_trials, batch_size):
        batch = class_data[i:i + batch_size]
        # Update covariance matrix
        cov += np.dot(batch.T, batch)

    # Normalize and add regularization
    cov /= n_class_trials
    cov += np.eye(cov.shape[0]) * 1e-6  # Add small regularization
    class_covs.append(cov)
    class_counts.append(n_class_trials)

"""### Calculate average covariance matrix incrementally"""
cov_avg = np.zeros_like(class_covs[0])
total_trials = sum(class_counts)

for cov, count in zip(class_covs, class_counts):
    cov_avg += cov * (count / total_trials)

"""### Add regularization to ensure matrix is invertible"""
cov_avg += np.eye(cov_avg.shape[0]) * 1e-6

"""### Calculate CSP filters for each class"""
csp_filters = []
for cov in class_covs:
    try:
        # Add regularization to both matrices
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
        cov_avg_reg = cov_avg + np.eye(cov_avg.shape[0]) * 1e-6
        
        # Ensure matrices are symmetric
        cov_reg = (cov_reg + cov_reg.T) / 2
        cov_avg_reg = (cov_avg_reg + cov_avg_reg.T) / 2
        
        # Solve generalized eigenvalue problem with regularization
        eigvals, eigvecs = eigh(cov_reg, cov_avg_reg)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        
        # Select top n_components
        csp_filters.append(eigvecs[:, :n_components])
    except np.linalg.LinAlgError as e:
        print(f"Warning: LinAlgError in CSP calculation: {e}")
        # Use identity matrix as fallback
        csp_filters.append(np.eye(n_channels * n_samples)[:, :n_components])

"""### Combine CSP filters from all classes"""
W = np.hstack(csp_filters)

"""### Apply CSP transformation in batches"""
csp_features = np.zeros((n_trials, W.shape[1]))
for i in range(0, n_trials, batch_size):
    batch = data_2d[i:i + batch_size]
    csp_features[i:i + batch_size] = np.dot(batch, W)

print("CSP feature extraction completed")

## Print feature information
print("\nFeature Information:")
print("CSP Features shape:", csp_features.shape)
print("Number of CSP components:", W.shape[1])

## Visualize CSP patterns
plt.figure(figsize=(12, 8))
n_plots = min(4, W.shape[1])
for i in range(n_plots):
    plt.subplot(2, 2, i+1)
    plt.plot(W[:, i])
    plt.title(f'CSP Pattern {i+1}')
plt.tight_layout()
plt.show()


"""## Save preprocessed data and features"""

np.save('preprocessed_data.npy', cleaned_data)
np.save('csp_features.npy', csp_features)

print('----------------------------------------')

"""# Model"""

"""## Load preprocessed data and features"""

cleaned_data = np.load('preprocessed_data.npy')
csp_features = np.load('csp_features.npy')

"""## Split data into training and testing sets"""

X_train, X_test, y_train, y_test = train_test_split(csp_features, y_labels, test_size=0.2, random_state=42)  # Use y_labels here too

## SVM Model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

## Evaluate model
y_pred = svm.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



"""## SVM Model"""
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

"""## Evaluate model"""
y_pred = svm.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

