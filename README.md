# EEG-Motor-Imagery-Based-Game

## Project Overview

This project implements a Brain-Computer Interface (BCI) system that uses EEG signals to control a Pacman game. The system processes EEG data from motor imagery tasks and uses machine learning to predict the intended movement.

## Implementation Details

- **Data Source**: [BCICIV_2a dataset](https://www.kaggle.com/datasets/aymanmostafa11/eeg-motor-imagery-bciciv-2a)
- **Signal Processing**: Bandpass and notch filtering, PCA, ICA, and CSP feature extraction
- **Model**: SVM, Random Forest, KNN, and LDA classifiers
- **Game**: Grid-based Pacman game with prediction and keyboard modes

## How to Try the Project

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/EEG-Motor-Imagery-Based-Game.git
   cd EEG-Motor-Imagery-Based-Game
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Model**:

   ```bash
   python Model_Test.py
   ```

4. **Play the Game**:
   ```bash
   python Pacman.py
   ```

## Current Progress and Challenges

- Model accuracy is around 30%
- Basic game mechanics are implemented with two modes:
     1. Prediction mode that makes Pacman play with the prediction moves of test data
     2. Normal mode for the user to play

## Future Improvements

- Enhance model accuracy
- Improve game features and user experience
