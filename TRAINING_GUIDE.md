# 🎓 Neural Network Training Guide - Where & How Data is Trained

## 📍 **Where is Training Data Generated?**

### **File Location:**
```
5x5TicTacTeo/
└── Backpropagation Algorithm/
    ├── game_logic.py          ← Training functions HERE
    ├── neural_network.py      ← Network architecture
    └── trained_model.pkl      ← Saved trained model
```

---
## 🔍 **Step-by-Step: Where Data Gets Trained**

### **1. Training Data Generation** 
**Location:** [game_logic.py](Backpropagation%20Algorithm/game_logic.py) - Line 245

```python
def generate_training_data(num_games: int = 100):
    """THIS IS WHERE TRAINING DATA IS CREATED"""
    X_train = []  # Will store board positions
    y_train = []  # Will store outcomes
    
    for game_num in range(num_games):  # Play N random games
        # Start a new game
        board = [[EMPTY_CELL] * 5 for _ in range(5)]
        game_state = GameState(board, None, PLAYER_X)
        game_history = []  # Store all moves
        
        # Play randomly until game ends
        while not game_state.is_game_over():
            moves = game_state.get_possible_moves()
            move = random.choice(moves)  # Random move
            
            # SAVE THIS BOARD STATE
            game_history.append((
                copy.deepcopy(game_state.board), 
                game_state.next_player
            ))
            
            game_state = game_state.apply_move(move)
        
        # Determine who won
        if has_player_won(game_state, PLAYER_O):
            outcome = 1.0   # AI won ✓
        elif has_player_won(game_state, PLAYER_X):
            outcome = -1.0  # AI lost ✗
        else:
            outcome = 0.0   # Draw −
        
        # CONVERT EACH BOARD TO TRAINING DATA
        for board_state, player in game_history:
            input_vector = board_to_vector(board_state)
            X_train.append(input_vector)  # The board
            
            if player == PLAYER_O:
                y_train.append([outcome])      # AI's perspective
            else:
                y_train.append([-outcome])     # Player's perspective
    
    return np.array(X_train), np.array(y_train)
```

**What happens here:**
1. ✅ Plays `num_games` random games (e.g., 2000 games)
2. ✅ Saves every board position from each game
3. ✅ Records final outcome (win/loss/draw)
4. ✅ Creates training pairs: `[board_position] → [outcome]`

**Example Training Data Created:**
```python
Game 1: AI Won (+1.0)
  Position 1: [-, -, -, -, -...] → +1.0
  Position 2: [X, -, -, -, -...] → +1.0
  Position 3: [X, O, -, -, -...] → +1.0
  ... (all 15 positions) → +1.0

Game 2: AI Lost (-1.0)
  Position 1: [-, -, -, -, -...] → -1.0
  Position 2: [X, -, -, -, -...] → -1.0
  ... → -1.0

Total: ~30,000 training examples from 2000 games
```

---

### **2. Training Function**
**Location:** [game_logic.py](Backpropagation%20Algorithm/game_logic.py) - Line 297

```python
def train_neural_network(num_games=500, epochs=100, save_path="trained_model.pkl"):
    """THIS IS WHERE TRAINING HAPPENS"""
    
    # STEP 1: Generate training data
    print(f"Generating training data from {num_games} random games...")
    X_train, y_train = generate_training_data(num_games)
    # X_train: 2D array of board positions (num_samples x 25)
    # y_train: 2D array of outcomes (num_samples x 1)
    
    # STEP 2: Initialize neural network
    if neural_network is None:
        initialize_ai()
    
    # STEP 3: Train the network
    print(f"Training neural network for {epochs} epochs...")
    neural_network.train(X_train, y_train, epochs=epochs, verbose=True)
    # This calls neural_network.py train() method
    
    # STEP 4: Save trained model to disk
    print(f"Saving trained model to {save_path}...")
    neural_network.save(save_path)
    print("Training complete!")
```

---

### **3. Actual Training (Backpropagation)**
**Location:** [neural_network.py](Backpropagation%20Algorithm/neural_network.py) - Line 150+

```python
def train(self, X, y, epochs=100, verbose=True):
    """THIS IS WHERE BACKPROPAGATION HAPPENS"""
    
    for epoch in range(epochs):  # Repeat N times
        total_error = 0
        
        # Process each training example
        for i in range(len(X)):
            # FORWARD PASS
            hidden = np.dot(X[i], self.weights1) + self.bias1
            hidden = self.relu(hidden)  # Activation
            
            output = np.dot(hidden, self.weights2) + self.bias2
            output = np.tanh(output)  # Final activation
            
            # CALCULATE ERROR
            error = y[i] - output
            total_error += abs(error)
            
            # BACKWARD PASS (THIS IS THE LEARNING!)
            # Calculate gradients
            output_gradient = error * self.tanh_derivative(output)
            hidden_gradient = np.dot(output_gradient, self.weights2.T)
            hidden_gradient *= self.relu_derivative(hidden)
            
            # UPDATE WEIGHTS (THIS MAKES THE NETWORK SMARTER)
            self.weights2 += self.learning_rate * np.outer(hidden, output_gradient)
            self.bias2 += self.learning_rate * output_gradient
            
            self.weights1 += self.learning_rate * np.outer(X[i], hidden_gradient)
            self.bias1 += self.learning_rate * hidden_gradient
        
        # Show progress
        if verbose and (epoch + 1) % 10 == 0:
            avg_error = total_error / len(X)
            print(f"Epoch {epoch+1}/{epochs}, Avg Error: {avg_error:.4f}")
```

---

### **4. Saved Model**
**Location:** `Backpropagation Algorithm/trained_model.pkl`

This file contains:
```python
{
    'weights1': array([[...], [...]]),  # 25x50 matrix
    'bias1': array([...]),              # 50 values
    'weights2': array([[...]]),         # 50x1 matrix
    'bias2': array([...]),              # 1 value
    'input_size': 25,
    'hidden_size': 50,
    'output_size': 1,
    'learning_rate': 0.01
}
```

---

## 🚀 **How to See Training in Action**

### **Option 1: Train from Command Line**

```bash
cd "Backpropagation Algorithm"
python game_logic.py train 1000 200
```

**You will see:**
```
Generating training data from 1000 random games...
Training neural network for 200 epochs...
Epoch 10/200, Avg Error: 0.7234
Epoch 20/200, Avg Error: 0.6189
Epoch 30/200, Avg Error: 0.5023
Epoch 40/200, Avg Error: 0.4127
...
Epoch 200/200, Avg Error: 0.0823
Saving trained model to trained_model.pkl...
Training complete!
```

### **Option 2: Train with Python Script**

Create `watch_training.py`:

```python
import sys
sys.path.append('Backpropagation Algorithm')

from game_logic import generate_training_data, initialize_ai
from neural_network import NeuralNetwork

# Initialize
print("Creating neural network...")
initialize_ai()

# Generate data
print("\n" + "="*50)
print("STEP 1: GENERATING TRAINING DATA")
print("="*50)
X_train, y_train = generate_training_data(num_games=500)
print(f"Generated {len(X_train)} training examples")
print(f"First example: {X_train[0][:10]}... → {y_train[0]}")

# Train
print("\n" + "="*50)
print("STEP 2: TRAINING NETWORK")
print("="*50)

# Show untrained prediction
from game_logic import neural_network
test_board = [[0]*5 for _ in range(5)]
before_score = neural_network.evaluate_board(test_board)
print(f"Before training - Empty board score: {before_score}")

# Train
neural_network.train(X_train, y_train, epochs=100, verbose=True)

# Show trained prediction
after_score = neural_network.evaluate_board(test_board)
print(f"\nAfter training - Empty board score: {after_score}")

# Save
print("\n" + "="*50)
print("STEP 3: SAVING MODEL")
print("="*50)
neural_network.save('my_trained_model.pkl')
print("Model saved to: my_trained_model.pkl")
```

Run it:
```bash
python watch_training.py
```

---

## 📊 **Training Data Flow Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│  1. PLAY RANDOM GAMES (game_logic.py line 245)              │
│                                                              │
│     Game 1: X-O-X-O-X-O → AI Won                           │
│     Game 2: O-X-X-O-O-X → Draw                             │
│     Game 3: X-X-O-O-X-X → Player Won                       │
│     ... (2000 games)                                        │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│  2. EXTRACT POSITIONS & OUTCOMES                            │
│                                                              │
│     Board State 1: [-, -, -, ...] → Outcome: +1.0          │
│     Board State 2: [X, -, -, ...] → Outcome: +1.0          │
│     Board State 3: [X, O, -, ...] → Outcome: +1.0          │
│     ... (30,000 examples)                                   │
│                                                              │
│     Stored in: X_train (boards), y_train (outcomes)        │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│  3. TRAIN NETWORK (neural_network.py train method)          │
│                                                              │
│     For each epoch (100 times):                             │
│       For each example (30,000):                            │
│         ├─ Forward pass: predict outcome                    │
│         ├─ Calculate error                                  │
│         ├─ Backward pass: compute gradients                 │
│         └─ Update weights                                   │
│                                                              │
│     Network gradually learns!                               │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│  4. SAVE TRAINED MODEL (neural_network.py save method)      │
│                                                              │
│     File: trained_model.pkl                                 │
│     Contains: weights1, weights2, bias1, bias2             │
│     Size: ~11 KB                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 **Inspect Current Training Data**

Check what's in your trained model:

```python
import pickle

# Load the model
with open('Backpropagation Algorithm/trained_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

print("Trained Model Contents:")
print(f"  Weights 1 shape: {model_data['weights1'].shape}")
print(f"  Weights 2 shape: {model_data['weights2'].shape}")
print(f"  Learning rate: {model_data['learning_rate']}")
print(f"  Network: {model_data['input_size']} → {model_data['hidden_size']} → {model_data['output_size']}")
```

---

## 🎯 **Summary: Where Training Happens**

| What | Where | Line |
|------|-------|------|
| **Data Generation** | `game_logic.py` | Line 245 |
| **Training Function** | `game_logic.py` | Line 297 |
| **Backpropagation** | `neural_network.py` | ~Line 150 |
| **Model Saving** | `neural_network.py` | ~Line 192 |
| **Saved Model** | `trained_model.pkl` | File in disk |

**Training Command:**
```bash
cd "Backpropagation Algorithm"
python game_logic.py train 2000 300
```

**What happens:**
1. 🎮 Plays 2000 random games
2. 📊 Creates ~30,000 training examples
3. 🧠 Trains network for 300 epochs
4. 💾 Saves to `trained_model.pkl`

Now the Neural Network knows how to play!
