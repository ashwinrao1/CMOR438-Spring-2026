# Perceptron — Wisconsin Breast Cancer Dataset

## Algorithm

The single-layer perceptron is a binary linear classifier that updates its weight vector
only when it makes an error: `w <- w + eta * y_i * x_i` for each misclassified sample.
The activation is the sign function; the decision boundary is the hyperplane `w'x = 0`.
The Perceptron Convergence Theorem guarantees convergence in finite iterations if the data
are linearly separable. There is no probabilistic output and no gradient of a smooth loss —
only a correction rule applied to misclassifications.

## Dataset

- **Source:** sklearn `load_breast_cancer` (Wisconsin Breast Cancer)
- **Task:** Binary classification — malignant (0) vs. benign (1)
- **Samples:** 569 (455 train / 114 test)
- **Features:** 30 continuous (cell nucleus measurements)
- **Class balance:** 212 malignant / 357 benign
- **Preprocessing:** StandardScaler fit on training data only

## Results

### Default Model

| Split | Accuracy |
|---|---|
| Train | 1.0000 |
| Test | **0.9386** |

### Learning Rate Comparison

| Learning rate | Test Accuracy | Weight norm ||w|| |
|---|---|---|
| 0.001 | 0.9386 | 0.1834 |
| 0.010 | 0.9386 | 1.8337 |
| 0.100 | 0.9386 | 18.3372 |
| 1.000 | 0.9386 | 183.3724 |

### 2D PCA Projection

| Split | Accuracy |
|---|---|
| Train | 0.9253 |
| Test | 0.9123 |

## Key Findings

**Performance and why:** The perceptron achieves 100% train accuracy and 93.9% test accuracy.
Training accuracy is perfect because the standardised 30-dimensional breast cancer features
are linearly separable — the perceptron is guaranteed to find a separating hyperplane and
stops updating once every training sample is correctly classified. Test accuracy drops to
93.9% because the separating hyperplane found by the perceptron does not necessarily maximise
the margin; it is simply the first hyperplane that classifies all training samples correctly,
and that hyperplane may not be maximally robust to the distributional shift between train
and test.

**Learning rate effect:** All four learning rates produce identical test accuracy (0.9386).
This is expected once the data are linearly separable: the perceptron converges to a valid
separator regardless of step size. The only observable difference is weight magnitude — the
norm scales exactly proportionally to the learning rate (0.1834, 1.8337, 18.337, 183.372 —
each a factor of 10 apart). A larger learning rate applies larger corrections per
misclassification, producing a geometrically equivalent boundary at a larger scale. Speed of
convergence was not measured here, but small rates require more epochs to reach the same
accuracy plateau.

**PCA decision boundary:** The 2D projection reduces test accuracy to 91.2% because two
principal components capture only a fraction of the 30-dimensional separating structure.
The boundary is a straight line in the projection — confirming the perceptron is always a
linear classifier — and the two classes separate well along the first principal component,
indicating that most discriminative variance is captured in the dominant eigenvector.

**Strengths of the architecture:** The perceptron is computationally efficient, requires no
matrix inversion or probabilistic assumptions, and converges provably on linearly separable
data. On the breast cancer dataset, where the standardised features are well-separated, it
achieves 93.9% test accuracy with a simple update rule.

**Limitations grounded in these results:** The 6% test error reveals the perceptron's
fundamental limitation: it finds any separating hyperplane, not the optimal one. A support
vector machine would maximise the margin and likely reduce test error further. More importantly,
the perceptron has no probabilistic output — it cannot express uncertainty or be calibrated
for a clinical threshold. It also has no convergence guarantee on non-separable data (updates
cycle indefinitely), and it produces no feature importance or coefficient interpretability
beyond the raw weight vector magnitudes.
