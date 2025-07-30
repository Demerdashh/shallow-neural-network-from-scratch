# 🧠 Shallow Neural Network from Scratch — Planar Data Classification

This project is a full implementation of a **2-layer neural network from scratch**, using **only NumPy**. It classifies **moons data** into two classes by manually building the entire training pipeline: initialization, forward/backward propagation, cost calculation, and gradient descent.

> **Inspired by** [deeplearning.ai’s Neural Networks and Deep Learning (Week 3)](https://www.coursera.org/learn/neural-networks-deep-learning) by Andrew Ng.

---

## 🚀 Features

- ✅ **Pure NumPy** — No ML libraries or frameworks used  
- ✅ Manual implementation of:
  - Forward propagation (ReLU + Sigmoid)
  - Cost computation (binary cross-entropy)
  - Backward propagation
  - Gradient descent optimization
- ✅ Vectorized implementation for efficiency
- ✅ Custom prediction and accuracy calculation
- ✅ Decision boundary visualization

---

## 🗂️ Folder Structure

<img width="321" height="516" alt="image" src="https://github.com/user-attachments/assets/19606554-a1a5-47e7-8ec8-d12141bde357" />


---

## 🧪 Dataset

- **Dataset**: moons dataset from `sklearn.datasets.make_moons()` 
- **Preprocessing**:
  - Normalized inputs
  - Binary labels (0 or 1)
  - Visualization to inspect decision boundary

---

## 📜 Methodology

This project is built fully from scratch — no `Keras`, no `scikit-learn`, no black-box training.

### 🔄 Preprocessing
- Generated or loaded 2D binary classification dataset
- Scaled and visualized

### 🧮 Model Architecture
- **Input layer**: 2 features (X1, X2)
- **Hidden layer**: Fully connected, with tanh activation
- **Output layer**: Sigmoid activation (for binary classification)

### 🧠 Training Pipeline
- Parameter initialization (He initialization)
- Forward pass (linear → tanh → linear → sigmoid)
- Compute cost using binary cross-entropy
- Backward propagation (manual gradients for tanh and sigmoid)
- Update parameters using gradient descent

### 📈 Evaluation
- Accuracy computed using custom logic
- Decision boundary visualized using `matplotlib`

---

## ❌ No Machine Learning Libraries Used

| Library        | Used? |
|----------------|-------|
| NumPy          | ✅     |
| scikit-learn   | ❌     |
| TensorFlow     | ❌     |
| Keras          | ❌     |
| PyTorch        | ❌     |

All logic is hand-coded to solidify understanding of forward and backward propagation.

---

## 📊 Sample Output

- Accuracy: **97.5%** on non-linearly separable data   
- Clear nonlinear decision boundary plotted

---

## 💡 Inspiration

This project is based on:

> 🎓 **deeplearning.ai – Neural Networks and Deep Learning (Week 3)**  
> by [Andrew Ng](https://www.andrewng.org/)

Additional self-imposed challenges:
- Used manual `matplotlib`-based decision boundary plotting  
- No reliance on any machine learning libraries  
- Used real activation gradients (tanh, sigmoid) implemented from scratch

---

## 🧑‍💻 Author

**Youssef Eldemerdash**  
_Learning deeply by building everything from scratch._

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for more details.

