# LSTM-based-Text-Classification-for-Sentiment-Analysis

## Project Overview
This project implements an LSTM neural network for sentiment analysis on the IMDb movie review dataset using TensorFlow/Keras.

## Requirements

### Software Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### Installation
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Or use the provided requirements.txt:
```bash
pip install -r requirements.txt
```

## File Structure
```
assignment4/
├── assignment.ipynb          # Main Jupyter notebook with all code
├── report.pdf                # Project report (1-2 pages)
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## How to Run

### Option 1: Google Colab (Recommended)
1. Upload `assignment.ipynb` to Google Colab
2. Run all cells sequentially from top to bottom
3. The notebook will:
   - Automatically download the IMDb dataset
   - Preprocess the data
   - Build and train the LSTM model
   - Generate evaluation metrics and visualizations
   - Perform error analysis

### Option 2: Local Jupyter Notebook
1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `assignment.ipynb`
4. Run all cells sequentially

### Option 3: Python Script
If you prefer running as a Python script:
1. Convert notebook to Python:
   ```bash
   jupyter nbconvert --to python assignment.ipynb
   ```
2. Run the script:
   ```bash
   python assignment.py
   ```

## Expected Runtime
- **On CPU:** Approximately 15-25 minutes
- **On GPU (Colab):** Approximately 5-10 minutes

## Output
The notebook will generate:
1. Data preprocessing statistics
2. Model architecture summary
3. Training history plots (accuracy and loss curves)
4. Test set evaluation metrics
5. Confusion matrix visualization
6. 5 misclassified examples with analysis

## Dataset
**IMDb Movie Review Dataset**
- 50,000 movie reviews (25,000 training, 25,000 testing)
- Binary classification: Positive (1) or Negative (0)
- Automatically downloaded via `tf.keras.datasets.imdb`

## Model Architecture
- Embedding Layer (20,000 vocab → 128 dimensions)
- LSTM Layer (128 units)
- Dropout Layer (0.5)
- Dense Layer (64 units, ReLU)
- Dropout Layer (0.3)
- Output Layer (1 unit, Sigmoid)

## Hyperparameters
- Vocabulary size: 20,000
- Maximum sequence length: 200
- Embedding dimension: 128
- Batch size: 64
- Epochs: 15 (with early stopping)
- Optimizer: Adam
- Loss function: Binary cross-entropy

## Expected Results
- Test Accuracy: ~85-87%
- Precision: ~0.85-0.87
- Recall: ~0.85-0.87
- F1-Score: ~0.85-0.87

## Troubleshooting

### Common Issues

**1. Out of Memory Error**
- Reduce batch size to 32
- Reduce max_len to 150

**2. Slow Training**
- Use Google Colab with GPU enabled (Runtime → Change runtime type → GPU)
- Reduce number of epochs

**3. Package Import Errors**
- Ensure all packages are installed: `pip install -r requirements.txt`
- Update TensorFlow: `pip install --upgrade tensorflow`

## Author
Umer Farooq Dar
omerfarooq2251@gmail.com

## Submission Date
January 17, 2026

## License
This project is created for academic purposes as part of Assignment 4.
