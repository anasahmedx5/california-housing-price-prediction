# California Housing Price Prediction

## Project Overview

This project focuses on building a predictive model to estimate housing prices in California based on various district-level features. It uses a feedforward neural network with two hidden layers to model complex relationships between features such as median income, average rooms, and population.

## Design Overview

### Data Preprocessing
- **Feature Selection**: The independent variables were extracted from the California Housing Dataset; the target was `MedianHouseValue`.
- **Scaling**: Applied `StandardScaler` to normalize the input features.
- **Train-Test Split**: Dataset was split into 67% training and 33% testing for evaluation.

### Model
- **Feedforward Neural Network**:
  - Input Layer: 8 features
  - Hidden Layer 1: 5 neurons with ReLU activation
  - Hidden Layer 2: 3 neurons with ReLU activation
  - Output Layer: 1 neuron (for regression output)

### Evaluation
- **Metrics**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
- **Visualization**:
  - Training vs Validation Loss plot to monitor model performance

## Key Components

### Libraries Used
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Scikit-learn**: Data preprocessing and loading
- **TensorFlow**: Model building, training, and evaluation

### Process
1. Load and explore the California Housing Dataset
2. Scale input features using `StandardScaler`
3. Split data into training and testing sets
4. Build and train a neural network using TensorFlow
5. Evaluate the model using MSE and MAE
6. Visualize training and validation loss over epochs

## Algorithms Used

- **Feedforward Neural Network**: Captures non-linear relationships in the data using hidden layers and ReLU activation.
- **MSE and MAE**: Quantify prediction error on the test set.

## Results

- The model achieved a **Mean Squared Error (MSE)** of `0.3684` and a **Mean Absolute Error (MAE)** of `0.4291` on the test set.
- Loss curves indicated good convergence without overfitting, thanks to careful architecture design and evaluation.
- The model effectively captured complex patterns in housing data, demonstrating the strength of neural networks for regression tasks.

This project highlights how neural networks can be leveraged for real-world regression problems involving housing price prediction.
