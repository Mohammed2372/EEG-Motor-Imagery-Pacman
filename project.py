# Libraries
## Data manipulation and analysis
import numpy as np
import pandas as pd

## Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

## Machine learning
from sklearn.model_selection import train_test_split

## signal processing
import mne
from sklearn.decomposition import PCA, FastICA
from mne.decoding import CSP
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt

# parameters
fs = 250

# Data
## old data overview
"""
there are E and T from every file like A01E and A01T, where E is the training set and T is the test set.
The data is divided into 9 sessions, each containing 288 trials. Each trial is 4 seconds long and contains 22 EEG channels. The data is sampled at 250 Hz, resulting in 1000 samples per trial. The labels are binary (1 or 2) and represent the two classes of motor imagery tasks: left hand and right hand.
"""
## load data
data = pd.read_csv('Data/eeg-motor-imagery-bciciv-2a/BCICIV_2a_all_patients.csv')

## data overview
data.head()
print('shape:', data.shape)
print('info:', data.info())
print('columns:', data.columns)
print('info:', data.info())
print('missing values:', data.isnull().sum())

# Preprocessing
## bandpass filter
def butter_bandpass(data, lowcut=8, highcut=30, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

## notch filter
def notch_filter(data, notch_freq):
    nyquist = 0.5 * fs
    low = notch_freq / nyquist
    b, a = butter(2, [low - 0.01, low + 0.01], btype='bandstop')
    return lfilter(b, a, data, axis=1)

## Common Average Reference (CAR)
def CAR(data):
    return data - np.mean(data, axis=0)

## remove blinking artifacts using ICA
def remove_artifacts_ica(data):
    info = mne.create_info(ch_names=data.shape[0], sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    ### ica
    ica = FastICA(n_components=0.95, random_state=42, max_iter='auto')
    ica.fit(raw)
    ### detect and remove eye blink
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    if eog_indices:
        ica.exclude = eog_indices
    
    clean_raw = ica.apply(raw.copy())
    return clean_raw.get_data()

## CSP
def extract_features_csp(data, labels, n_components=4):
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    csp.fit_transform(data, labels)
    return csp

## preprocessing pipeline
def preprocess_trial(trial):
    filtered_data = butter_bandpass(trial)
    notch_filtered = notch_filter(filtered_data)
    car_filtered = CAR(notch_filtered)
    clean_data = remove_artifacts_ica(car_filtered)
    normalized_data = (clean_data - np.mean(clean_data)) / np.std(clean_data)
    return normalized_data

## process data
def process_data(x, y):
    processed_trials = []
    
    for i in x:
        processed_trials.append(preprocess_trial(i))
    
    processed_data = np.array(processed_trials)
    
    ### extract features
    features = extract_features_csp(processed_data, y)
    
    return features
