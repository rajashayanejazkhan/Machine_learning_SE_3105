# Classification of MNIST Handwritten Digits Using Machine Learning

## Introduction
The MNIST dataset consists of grayscale images of handwritten digits (0-9), each represented as a 28x28 pixel image. The dataset has been preprocessed into CSV format, with each image flattened into a 1D vector of 784 features. The objective of this study is to experiment with different classification models, evaluate their performance, and determine the most effective approach for digit recognition.

## Methodology
### Data Preparation
- Loaded `mnist_train.csv` and `mnist_test.csv` into Pandas DataFrames.
- Separated features and labels.
- Normalized pixel values to range [0,1] to enhance model performance.

### Models Used
1. **Logistic Regression** (max_iter=500)
2. **K-Nearest Neighbors (K=5)**
3. **Naïve Bayes**

### Hyperparameter Tuning
- Used `k=5` for KNN.
- Applied smoothing techniques in Naïve Bayes.
- Adjusted iterations for Logistic Regression to ensure convergence.

## Results
### Model Performance Comparison
| Model               | Accuracy (Train) | Accuracy (Test) |
|--------------------|----------------|---------------|
| Logistic Regression | 93.5%          | 92.7%         |
| KNN (k=5)         | 96.2%          | 95.4%         |
| Naïve Bayes       | 85.3%          | 84.9%         |

### Visualization
- Accuracy comparison graph.
- Confusion matrices for each model.
- Misclassified samples analysis.

### Graphical Representation
![image](https://github.com/user-attachments/assets/a2b94ff3-c3bc-4659-890f-b5bb50606a5a)


## Discussion
- **KNN performed best**, achieving 95.4% accuracy on test data due to its ability to capture complex decision boundaries.
- **Logistic Regression performed well**, but slightly lagged behind KNN as it assumes a linear decision boundary.
- **Naïve Bayes had the lowest performance**, likely due to its assumption of feature independence, which is not entirely true for MNIST.
- Normalization and hyperparameter tuning played crucial roles in improving performance.

## Conclusion
- **KNN (k=5) is the best performing model for MNIST classification**.
- Logistic Regression is a good alternative with lower computational cost.
- Naïve Bayes is fast but less accurate.
- Future improvements could include **deep learning approaches (e.g., CNNs) for even better accuracy**.
