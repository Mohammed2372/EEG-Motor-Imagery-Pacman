"""# Libraries"""
## Data manipulation and analysis
import numpy as np
import pandas as pd
from joblib import dump

## Data visualization
import matplotlib.pyplot as plt

## signal processing
from sklearn.decomposition import PCA, FastICA
from mne.decoding import CSP
from scipy import signal
from scipy.signal import butter, filtfilt

## Machine learning
from sklearn.model_selection import GridSearchCV  # Removed train_test_split as we're doing manual splitting
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
samples_per_trial = 201  # Each epoch/trial has 201 samples

"""# Data

## old data overview

- there are E and T from every file like A01E and A01T, where E is the training set and T is the test set.
- The data is divided into 9 sessions, each containing 288 trials. Each trial is 4 seconds long and contains 22 EEG channels.
- The data is sampled at 250 Hz, resulting in 1000 samples per trial.
- The labels are binary (1, 2, 3, 4) and represent the two classes of motor imagery tasks: left hand, right hand, Tounge and Foot.
"""

"""## load data"""

data = pd.read_csv('Data/eeg-motor-imagery-bciciv-2a/BCICIV_2a_all_patients.csv')

# Add epoch column if not present
if 'epoch' not in data.columns:
    # Calculate epochs based on the fact that each trial has 201 samples
    data['epoch'] = data.index // samples_per_trial

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

# Get all EEG columns (sorted to ensure consistent order)
eeg_columns = sorted([col for col in data.columns if col.startswith('EEG-')])
n_channels = len(eeg_columns)  # This should be 22 EEG channels

print("\nEEG channels being used:", eeg_columns)
print("Number of EEG channels found:", n_channels)

# Group data first by patient, then by epoch to get trials
print("\nProcessing data by patient and epoch...")
X_reshaped_all = []
y_labels_all = []

# Group by patient first
for patient, patient_data in data.groupby('patient'):
    print(f"\nProcessing patient {patient}")
    
    # Then group by epoch within each patient's data
    patient_grouped = patient_data.groupby('epoch')
    patient_n_trials = len(patient_grouped)
    print(f"Number of trials for patient {patient}: {patient_n_trials}")
    
    # Initialize arrays for this patient's reshaped data
    patient_X = np.zeros((patient_n_trials, n_channels, samples_per_trial))
    patient_y = np.zeros(patient_n_trials)
    
    # Process each epoch for this patient
    for i, (epoch, epoch_data) in enumerate(patient_grouped):
        # Get EEG data for this epoch
        epoch_eeg = epoch_data[eeg_columns].values
        
        if epoch_eeg.shape[0] != samples_per_trial:
            print(f"Warning: Epoch {epoch} has {epoch_eeg.shape[0]} samples instead of {samples_per_trial}")
            continue
            
        # Transpose to get (channels, samples)
        epoch_eeg_T = epoch_eeg.T
        patient_X[i] = epoch_eeg_T
        
        # Store the label (they're all the same within an epoch)
        patient_y[i] = epoch_data['label'].iloc[0]
    
    # Append this patient's data to the main arrays
    X_reshaped_all.append(patient_X)
    y_labels_all.append(patient_y)

# Combine all patients' data
X_reshaped = np.concatenate(X_reshaped_all, axis=0)
y_labels = np.concatenate(y_labels_all)

print("\nFinal data shapes:")
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

# Save preprocessed data after filtering
np.save('Saved Data/filtered_data.npy', filtered_data)
print("\nFiltered data saved successfully")

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

# Save PCA data
np.save('Saved Data/pca_data.npy', pca_data)
print("\nPCA data saved successfully")

print(f"\nPCA data shape: {pca_data.shape}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.2%}")

print("\nData shape after PCA:", pca_data.shape)
print("Class distribution after PCA:")
unique_labels, counts = np.unique(y_labels, return_counts=True)

for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} trials")

# print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())

"""### Plot PCA components"""
# Get column names (features)
feature_names = data.drop('label', axis=1).columns.tolist()

# Plot each column's contribution to PCA components separately
# print("\nGenerating individual plots for each feature's PCA contributions...")
# for col_idx, col_name in enumerate(feature_names):
#     plt.figure(figsize=(6, 2))  # Made even smaller for better visibility
#     # Get contribution of this column to each PCA component
#     contributions = [pca.components_[i][col_idx] for i in range(n_components_pca)]
#     plt.bar(range(n_components_pca), contributions)
#     plt.title(f'{col_name}', fontsize=8)
#     plt.xlabel('PCA Component')
#     plt.ylabel('Weight')
#     plt.xticks(range(n_components_pca), [f'{i+1}' for i in range(n_components_pca)], fontsize=6)
#     plt.tight_layout()
#     plt.show()
#     plt.close()  # Close the figure to free memory
    
# # Plot variance explained by each component
# plt.figure(figsize=(6, 2))
# plt.bar(range(n_components_pca), pca.explained_variance_ratio_)
# plt.title('Variance Explained by Components', fontsize=8)
# plt.xlabel('Component')
# plt.ylabel('Ratio')
# plt.xticks(range(n_components_pca), [f'{i+1}' for i in range(n_components_pca)], fontsize=6)
# plt.tight_layout()
# plt.show()
# plt.close()


# # Plot time series for each component separately
# print("\nGenerating time series plots for each PCA component...")
# for i in range(n_components_pca):
#     plt.figure(figsize=(6, 2))
#     component_data = pca_data[:, i, :].flatten()
#     plt.plot(component_data, alpha=0.7, linewidth=0.5)
    
#     # Add top contributing features
#     loadings = abs(pca.components_[i])
#     top_features_idx = loadings.argsort()[-3:][::-1]
#     top_features = [feature_names[idx][:10] for idx in range(n_components_pca)]
    
#     plt.title(f'PC{i+1} Time Series\nVar: {pca.explained_variance_ratio_[i]:.2%}\nTop: {", ".join(top_features)}', 
#               fontsize=8)
#     plt.tight_layout()
#     plt.show()
#     plt.close()

"""## apply ICA"""

print("\nApplying ICA...")
# Reshape for ICA while preserving temporal information
ica_input = pca_data.reshape(n_trials * n_samples, n_components_pca)
ica = FastICA(n_components=n_components_ica, random_state=42)
ica_data = ica.fit_transform(ica_input)

# Save ICA data
np.save('Saved Data/ica_data.npy', ica_data)
print("\nICA data saved successfully")

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

fig, ax = plt.subplots(4, 4, figsize=(12, 12))  # Adjusted for 16 columns
row = 0
col = 0
count = 0

# Assuming you have the column names in a list called `column_names`
column_names = [f"Column {i+1}" for i in range(n_components_ica)]  # Replace with actual column names if available

for i in range(n_components_ica):
    ax[row, col].plot(ica_data[:, i]**2)
    ax[row, col].set_title(column_names[i])
    count += 1
    col += 1
    if count % 4 == 0:
        row += 1
        col = 0

for ax in fig.get_axes():
    ax.label_outer()

fig.tight_layout()

"""remove components with blink in it"""
components_to_remove = []

# Create mask for components to keep
keep_mask = np.ones(n_components_ica, dtype=bool)
keep_mask[components_to_remove] = False
remaining_components = np.where(keep_mask)[0]

# Keep only the desired components
ica_data = ica_data[:, remaining_components]

print(f"Removed components: {components_to_remove}")
print(f"Remaining components: {len(remaining_components)}")
print(f"ICA data shape before reshaping: {ica_data.shape}")

"""### reshape ICA data"""
# Calculate dimensions
n_timepoints = ica_data.shape[0]
n_remaining = len(remaining_components)
calculated_trials = n_timepoints // n_samples

print("Calculated dimensions:")
print(f"Total timepoints: {n_timepoints}")
print(f"Samples per trial: {n_samples}")
print(f"Calculated number of trials: {calculated_trials}")
print(f"Number of remaining components: {n_remaining}")

# Reshape back to maintain temporal structure
ica_data_reshaped = ica_data.reshape(calculated_trials, n_samples, n_remaining)
# Transpose to get (trials, channels, samples)
ica_data_reshaped = np.transpose(ica_data_reshaped, (0, 2, 1))

print(f"ICA data shape after reshaping: {ica_data_reshaped.shape}")

# # Verify dimensions
# expected_components = len(remaining_components)
# if (ica_data_reshaped.shape[1] == expected_components and 
#     ica_data_reshaped.shape[0] == calculated_trials and 
#     ica_data_reshaped.shape[2] == n_samples):
#     print("\nDimension verification successful:")
#     print(f"- Number of trials: {ica_data_reshaped.shape[0]}")
#     print(f"- Number of components: {ica_data_reshaped.shape[1]} (expected {expected_components})")
#     print(f"- Samples per trial: {ica_data_reshaped.shape[2]} (expected {n_samples})")
# else:
#     print("\nWarning: Dimension mismatch:")
#     print(f"- Current shape: {ica_data_reshaped.shape}")
#     print(f"- Expected: (n_trials={calculated_trials}, n_components={expected_components}, n_samples={n_samples})")

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
for class_id in range(1, 5):
    n_trials = np.sum(y_labels == class_id)
    print(f"Number of trials for class {class_id}: {n_trials}")

# Additional check to make sure data and labels align
assert len(cleaned_data) == len(y_labels), "Number of trials doesn't match number of labels"
assert not np.any(np.isnan(cleaned_data)), "Data contains NaN values"
assert not np.any(np.isinf(cleaned_data)), "Data contains infinite values"

# Fit and transform the data
csp_features = csp.fit_transform(cleaned_data, y_labels)

# Save CSP features
np.save('Saved Data/csp_features.npy', csp_features)
print("\nCSP features saved successfully")

print("CSP feature extraction completed")
print("\nFeature Information:")
print("CSP Features shape:", csp_features.shape)

# Save the CSP patterns for visualization
np.save('Saved Data/csp_patterns.npy', csp.patterns_)
print("CSP patterns saved successfully")

# Save the labels
np.save('Saved Data/y_labels.npy', y_labels)
print("Labels saved successfully")

"""# Visualizing CSP patterns"""
class_names = ['Right Hand', 'Left Hand', 'Feet', 'Tongue']  # Updated class names for all 4 classes
# print("\nVisualizing CSP patterns...")
# patterns = csp.patterns_
# patterns_norm = patterns / np.max(np.abs(patterns))

# plt.figure(figsize=(16, 4))
# for i in range(min(4, n_components_csp)):  # Show patterns for all 4 classes
#     plt.subplot(1, 4, i + 1)
#     plt.imshow(patterns_norm[i].reshape(4, 4), cmap='jet', interpolation='nearest')
#     plt.title(f'CSP Pattern {i+1}')
#     plt.colorbar()
# plt.tight_layout()
# plt.show()

"""## Save preprocessed data and features"""

np.save('Saved Data/preprocessed_data.npy', cleaned_data)


print("\nPreprocessed data and features saved successfully")

"""# Model"""

"""## Load preprocessed data and features"""

cleaned_data = np.load('Saved Data/preprocessed_data.npy')
csp_features = np.load('Saved Data/csp_features.npy')

"""## Split data into training and testing sets per patient"""

# Get patient IDs from the original data
patient_ids = data['patient'].unique()
n_patients = len(patient_ids)
n_classes = 4
trials_per_class = 72
train_trials_per_class = 60
test_trials_per_class = 12

# Initialize arrays for final train and test sets
X_train = []
X_test = []
y_train = []
y_test = []

# Calculate trials per patient
trials_per_patient = len(csp_features) // n_patients
print(f"\nTotal features: {len(csp_features)}")
print(f"Number of patients: {n_patients}")
print(f"Trials per patient: {trials_per_patient}")

print("\nSplitting data by patient and class:")
for i, patient_id in enumerate(patient_ids):
    print(f"\nProcessing patient {patient_id}:")
    
    # Get this patient's portion of the data
    start_idx = i * trials_per_patient
    end_idx = (i + 1) * trials_per_patient
    patient_features = csp_features[start_idx:end_idx]
    patient_labels = y_labels[start_idx:end_idx]
    
    print(f"Patient data shape: {patient_features.shape}")
    print(f"Unique labels for patient: {np.unique(patient_labels, return_counts=True)}")
      # Split by class
    for class_id in range(1, n_classes + 1):
        class_mask = patient_labels == class_id
        class_features = patient_features[class_mask]
        
        print(f"  Class {class_id}:")
        print(f"    Number of trials: {len(class_features)}")
        
        # Skip if we don't have any trials for this class
        if len(class_features) == 0:
            print(f"    Warning: No trials found for patient {patient_id}, class {class_id}")
            continue
            
        # Split into train and test (using all available trials)
        n_trials = len(class_features)
        n_train = min(train_trials_per_class, int(0.8 * n_trials))  # Use 80% for training if we have fewer trials
        n_test = min(test_trials_per_class, n_trials - n_train)
        
        print(f"    Using {n_train} trials for training, {n_test} for testing")
        
        X_train.append(class_features[:n_train])
        if n_test > 0:
            X_test.append(class_features[n_train:n_train + n_test])
            y_train.extend([class_id] * n_train)
            y_test.extend([class_id] * n_test)

# Convert to numpy arrays
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("\nTraining set:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print("\nClass distribution in training set:")
unique_labels, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples (expected {train_trials_per_class * n_patients} samples)")

print("\nTest set:")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("\nClass distribution in test set:")
unique_labels, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples (expected {test_trials_per_class * n_patients} samples)")

# Verify balance across patients and classes
print("\nVerifying data balance:")
print(f"Expected per patient per class: {train_trials_per_class} train, {test_trials_per_class} test")
print(f"Expected total per class: {train_trials_per_class * n_patients} train, {test_trials_per_class * n_patients} test")
print(f"Total samples: {len(y_train)} train, {len(y_test)} test")

"""## Test Multiple Models"""
print("\nTesting Multiple Models...")

# Initialize models
models = {
    'SVM': SVC(C=100, gamma=0.1, kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LDA': LinearDiscriminantAnalysis()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred, target_names=class_names)
    }
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"\n{name} Classification Report:")
    print(results[name]['report'])

# Save only the SVM model as it's our main model
dump(models['SVM'], 'svm_model.joblib')
print("\nSVM model saved successfully")

"""## SVM Model with Grid Search"""
# # Define parameter grid
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'degree': [2, 3, 4]  # Only used by poly kernel
# }

# print("\nPerforming Grid Search for SVM parameters...")
# grid_search = GridSearchCV(
#     SVC(random_state=42),
#     param_grid,
#     cv=5,
#     n_jobs=-1,  # Use all available cores
#     verbose=2,
#     scoring='accuracy'
# )

# # Fit grid search
# grid_search.fit(X_train, y_train)

# # Print results
# print("\nGrid Search Results:")
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

"""## SVM model with best parameters"""
# Get the best model
svm = SVC(C=100, gamma=0.1, kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
# Evaluate on test set
y_pred = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest set accuracy: {test_accuracy:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# #Print parameter ranking
# print("\nParameter importance ranking:")
# cv_results = pd.DataFrame(grid_search.cv_results_)
# for i, params in enumerate(cv_results['params']):
#     mean_score = cv_results['mean_test_score'][i]
#     std_score = cv_results['std_test_score'][i]
#     print(f"Parameters: {params}")
#     print(f"Mean CV Score: {mean_score:.4f} (+/- {std_score*2:.4f})\n")

"""## Test Additional Models"""
print("\nTesting Additional Models...")

# Initialize additional models
additional_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LDA': LinearDiscriminantAnalysis()
}

# Train and evaluate each additional model
additional_results = {}
for name, model in additional_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    additional_results[name] = {
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred, target_names=class_names)
    }
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"\n{name} Classification Report:")
    print(additional_results[name]['report'])

"""## Save test data and model for later evaluation"""
np.save('Saved Data/X_test.npy', X_test)
np.save('Saved Data/y_test.npy', y_test)
print("Test data saved successfully")
dump(svm, 'svm_model.joblib')
print("model saved successfully")
