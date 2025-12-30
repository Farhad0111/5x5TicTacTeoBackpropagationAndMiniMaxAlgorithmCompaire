import numpy as np
import pickle
import os
from typing import List, Tuple

class NeuralNetwork:
    """Neural network with backpropagation for Tic-Tac-Toe evaluation."""
    
    def __init__(self, input_size: int = 25, hidden_size: int = 50, output_size: int = 1, 
                 learning_rate: float = 0.01):
        """
        Initialize neural network.
        
        Args:
            input_size: Size of input layer (25 for 5x5 board)
            hidden_size: Size of hidden layer
            output_size: Size of output layer (1 for board evaluation)
            learning_rate: Learning rate for backpropagation
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases with Xavier initialization
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.bias1 = np.zeros((1, hidden_size))
        
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.bias2 = np.zeros((1, output_size))
        
        # For momentum
        self.velocity_w1 = np.zeros_like(self.weights1)
        self.velocity_b1 = np.zeros_like(self.bias1)
        self.velocity_w2 = np.zeros_like(self.weights2)
        self.velocity_b2 = np.zeros_like(self.bias2)
        self.momentum = 0.9
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)
    
    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - x ** 2
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward propagation.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (hidden layer input, hidden layer output, output layer input, final output)
        """
        # Hidden layer
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.tanh(self.z2)  # Use tanh for output [-1, 1]
        
        return self.z1, self.a1, self.z2, self.a2
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """
        Backward propagation with gradient descent.
        
        Args:
            X: Input data
            y: True labels
            output: Network output from forward pass
        """
        m = X.shape[0]
        
        # Calculate gradients
        # Output layer
        dz2 = output - y
        dw2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer
        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self.relu_derivative(self.a1)
        dw1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights with momentum
        self.velocity_w2 = self.momentum * self.velocity_w2 + self.learning_rate * dw2
        self.velocity_b2 = self.momentum * self.velocity_b2 + self.learning_rate * db2
        self.velocity_w1 = self.momentum * self.velocity_w1 + self.learning_rate * dw1
        self.velocity_b1 = self.momentum * self.velocity_b1 + self.learning_rate * db1
        
        self.weights2 -= self.velocity_w2
        self.bias2 -= self.velocity_b2
        self.weights1 -= self.velocity_w1
        self.bias1 -= self.velocity_b1
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = False):
        """
        Train the neural network.
        
        Args:
            X: Training data
            y: Training labels
            epochs: Number of training epochs
            verbose: Whether to print progress
        """
        for epoch in range(epochs):
            # Forward propagation
            _, _, _, output = self.forward(X)
            
            # Calculate loss (Mean Squared Error)
            loss = np.mean((output - y) ** 2)
            
            # Backward propagation
            self.backward(X, y, output)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        _, _, _, output = self.forward(X)
        return output
    
    def evaluate_board(self, board: List[List[str]]) -> float:
        """
        Evaluate a board position.
        
        Args:
            board: 5x5 board state
            
        Returns:
            Evaluation score between -1 and 1
        """
        # Convert board to input vector
        input_vector = self.board_to_vector(board)
        input_array = np.array([input_vector])
        
        # Get prediction
        score = self.predict(input_array)[0][0]
        return float(score)
    
    def board_to_vector(self, board: List[List[str]]) -> List[float]:
        """
        Convert board to input vector for neural network.
        
        Args:
            board: 5x5 board state
            
        Returns:
            Flattened vector representation
        """
        vector = []
        for row in board:
            for cell in row:
                if cell == 'X':
                    vector.append(1.0)
                elif cell == 'O':
                    vector.append(-1.0)
                else:
                    vector.append(0.0)
        return vector
    
    def save(self, filename: str):
        """Save the neural network to a file."""
        data = {
            'weights1': self.weights1,
            'bias1': self.bias1,
            'weights2': self.weights2,
            'bias2': self.bias2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filename: str):
        """Load the neural network from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.weights1 = data['weights1']
                self.bias1 = data['bias1']
                self.weights2 = data['weights2']
                self.bias2 = data['bias2']
                self.input_size = data['input_size']
                self.hidden_size = data['hidden_size']
                self.output_size = data['output_size']
                self.learning_rate = data['learning_rate']
            return True
        return False
