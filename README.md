# Handwritten Digit Recognition - Virtualyyst Assignment

## Overview
This project is focused on developing a machine learning pipeline to classify grayscale images of handwritten digits (0-9) from the MNIST dataset. The project implements and compares three classical machine learning models—K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree—implemented from scratch and using scikit-learn.

## Project Structure
- `main.ipynb`: The main Jupyter Notebook containing the entire pipeline: data loading, preprocessing, model implementation, evaluation, and reporting.
- `mnist_train.csv`: Training dataset (MNIST subset).
- `mnist_test.csv`: Testing dataset (MNIST subset).


## Requirements
The project requires Python 3 and the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Installation
1. Clone the repository or navigate to the project directory.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   # .venv\Scripts\activate   # On Windows
   ```
3. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```

## Workflow
The notebook `main.ipynb` follows these steps:
1.  **Data Loading & Exploration:** Loading MNIST CSV data, checking statistics, and visualizing sample images.
2.  **Preprocessing:** Normalizing pixel values (0-1) and splitting data into training and testing sets.
3.  **Model Implementation:**
    *   **K-Nearest Neighbors (KNN):** Implemented with `k=3`.
    *   **Support Vector Machine (SVM):** Implemented with a linear kernel.
    *   **Decision Tree:** hyperparameter tuning for `max_depth` and `min_samples_split`.
4.  **Evaluation:** Calculating accuracy and generating confusion matrices.
5.  **Reporting:** Comparative analysis of model performance.

## Results
| Model | Accuracy | Key Observations |
| :--- | :--- | :--- |
| **K-Nearest Neighbors (KNN)** | **~97.05%** | Best performer. Effectively captures local non-linear patterns in handwriting (k=3). |
| **Support Vector Machine (SVM)** | ~91.70% | Good performance with Linear Kernel but computationally expensive. |
| **Decision Tree** | ~87.9% | Lowest accuracy. Struggled to generalize on pixel-level data compared to distance-based metrics. |

## Insights
- **Best Model:** KNN performed the best because the data forms tight clusters suitable for distance-based classification.
- **Misclassifications:** Errors mostly occurred between structurally similar digits, such as:
    - 4 vs 9
    - 3 vs 5
    - 6 vs 8
    - 1 vs 7
- **Future Improvements:**
    - Using **Convolutional Neural Networks (CNNs)** would significantly improve feature extraction.
    - Applying **Dimensionality Reduction (PCA)** can reduce computation costs without sacrificing much accuracy.

## Usage
To replicate the results, open the `main.ipynb` notebook in Jupyter and run all cells sequentially.
```bash
jupyter notebook main.ipynb
```
