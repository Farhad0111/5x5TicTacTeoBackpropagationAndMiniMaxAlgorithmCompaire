# 5x5 Tic-Tac-Toe Game with AI 🎮

A complete 5x5 Tic-Tac-Toe game with beautiful GUI featuring two different AI implementations: MiniMax Algorithm and Neural Network with Backpropagation.

## 🚀 Quick Start

### Running the Game

Simply run the main launcher:

```bash
python main_launcher.py
```

Or run the GUI directly:

```bash
python gui_launcher.py
```

## 🎯 Game Features

### Main Menu
- **Intuitive GUI** with easy game mode selection
- **Beautiful Design** with color-coded elements
- **Two AI Options** to choose from

### Game Modes

#### 1. 🤖 MiniMax Algorithm
- Classic AI using MiniMax algorithm with Alpha-Beta pruning
- Perfect strategic play
- Challenging opponent that thinks several moves ahead
- Fast decision making

#### 2. 🧠 Neural Network (Backpropagation)
- AI trained with Neural Network using Backpropagation
- Learning-based approach
- Adaptive gameplay
- Improves with training

## 🎲 How to Play

1. **Launch the Game**: Run `python main_launcher.py`
2. **Select Game Mode**: Choose between MiniMax AI or Neural Network AI
3. **Play**: 
   - You are **X** (Red)
   - AI is **O** (Blue)
   - Click on any empty cell to make your move
   - Get **3 in a row** (horizontal, vertical, or diagonal) to win!
4. **New Game**: Click "New Game" button to restart
5. **Back to Menu**: Return to main menu to select a different AI

## 📋 Requirements

- Python 3.7 or higher
- tkinter (usually included with Python)
- numpy (for Neural Network mode)

### Installing Dependencies

```bash
pip install numpy
```

For the Neural Network mode, you can also install additional dependencies:

```bash
cd "Backpropagation Algorithm"
pip install -r requirements.txt
```

## 🎨 GUI Features

### Main Menu
- Clean and modern design
- Easy-to-understand game mode descriptions
- Color-coded options (Green for MiniMax, Blue for Neural Network)

### Game Board
- 5x5 grid with clear cell boundaries
- Visual feedback for moves
- Color-coded pieces (Red for player, Blue for AI)
- Status messages showing whose turn it is
- Win/Loss/Tie notifications

### Controls
- **New Game**: Start a fresh game with the same AI
- **Back to Menu**: Return to mode selection screen
- **Exit**: Close the application

## 🧠 How the AI Algorithms Work

### 🤖 MiniMax Algorithm (Game Tree Search)

**Location:** `MiniMax Algorithm/FinalTicTacTeo.py`

#### **What is MiniMax?**
MiniMax is a classic game theory algorithm that assumes both players play optimally. It explores the game tree to find the best move by simulating all possible future moves.

#### **How It Works:**

1. **Game Tree Exploration**
   ```
   Current Position
   ├─ Move 1 (AI)
   │  ├─ Move A (Player)
   │  │  ├─ Move X (AI) → Score: +5
   │  │  └─ Move Y (AI) → Score: +3
   │  └─ Move B (Player)
   │     ├─ Move X (AI) → Score: -2
   │     └─ Move Y (AI) → Score: +1
   └─ Move 2 (AI)
      └─ ... (continues)
   ```

2. **Recursive Evaluation**
   - **Maximizing Player (AI/O):** Tries to maximize the score
   - **Minimizing Player (Human/X):** Tries to minimize the score
   - Recursively evaluates all possible game states up to a certain depth

3. **Alpha-Beta Pruning Optimization**
   - Cuts off branches that won't affect the final decision
   - Significantly reduces the number of positions evaluated
   - Makes the algorithm much faster without losing accuracy
   
   Example:
   ```
   If we already found a move with score +5,
   and a branch shows it can only get worse (< +5),
   we stop exploring that branch (prune it).
   ```

4. **Position Scoring**
   - **Win (AI):** +100 + depth (favors faster wins)
   - **Loss (AI):** -100 - depth (avoids faster losses)
   - **Draw:** 0
   - **Ongoing:** Evaluates based on potential winning patterns

5. **Move Selection**
   - AI always picks the move with the highest evaluated score
   - Guarantees optimal play within search depth
   - Very difficult to beat!

#### **Strengths:**
- ✅ Deterministic and consistent
- ✅ Plays optimally within search depth
- ✅ No training required
- ✅ Guaranteed to find winning moves

#### **Limitations:**
- ⚠️ Computationally expensive for deep searches
- ⚠️ Limited by search depth (default: 4 moves ahead)
- ⚠️ Performance degrades on larger boards

---

### 🧠 Neural Network with Backpropagation (Machine Learning)

**Location:** `Backpropagation Algorithm/`

#### **What is a Neural Network?**
A Neural Network is a machine learning model inspired by the human brain. It learns patterns from data through training and can make decisions based on what it has learned.

#### **Network Architecture:**

```
Input Layer (25 neurons)
    │  Each neuron = one cell on 5x5 board
    │  Values: 1 (AI/O), -1 (Player/X), 0 (Empty)
    ↓
Hidden Layer (50 neurons)
    │  With ReLU activation: f(x) = max(0, x)
    │  Learns complex patterns and strategies
    ↓
Output Layer (1 neuron)
    │  With Tanh activation: f(x) = (e^x - e^-x)/(e^x + e^-x)
    │  Output: Board evaluation score (-1 to +1)
    └→ Higher = Better for AI, Lower = Better for Player
```

#### **How It Works:**

1. **Board Representation**
   ```python
   5x5 Board:        Flattened Input Vector:
   [X, O, -, -, -]   [−1, 1, 0, 0, 0,
   [-, X, O, -, -]    0, −1, 1, 0, 0,
   [-, -, X, -, -] →  0, 0, −1, 0, 0,
   [O, -, -, -, -]    1, 0, 0, 0, 0,
   [-, -, -, -, O]    0, 0, 0, 0, 1]
   ```

2. **Forward Pass (Making a Prediction)**
   ```
   Step 1: Input → Hidden Layer
   hidden = ReLU(input × weights1 + bias1)
   
   Step 2: Hidden → Output Layer
   output = Tanh(hidden × weights2 + bias2)
   
   Result: Score indicating how good the position is
   ```

3. **Move Selection**
   - Evaluates all possible moves
   - Each move creates a new board state
   - Forward pass computes score for each state
   - Selects move with highest score
   - Also checks for immediate wins and blocks opponent wins

4. **Backpropagation (Learning Process)**
   
   When training, the network learns from game outcomes:
   
   ```
   ┌─────────────────────────────────────┐
   │  1. Play a game and get result      │
   │     Win: +1, Loss: -1, Draw: 0      │
   └──────────────┬──────────────────────┘
                  ↓
   ┌─────────────────────────────────────┐
   │  2. Calculate Error                 │
   │     Error = Actual - Predicted      │
   └──────────────┬──────────────────────┘
                  ↓
   ┌─────────────────────────────────────┐
   │  3. Backward Pass                   │
   │     Compute gradient for each weight│
   │     using chain rule                │
   └──────────────┬──────────────────────┘
                  ↓
   ┌─────────────────────────────────────┐
   │  4. Update Weights                  │
   │     weight = weight - (learning_rate│
   │              × gradient)            │
   └──────────────┬──────────────────────┘
                  ↓
   ┌─────────────────────────────────────┐
   │  5. Repeat for many games           │
   │     Network gradually improves!     │
   └─────────────────────────────────────┘
   ```

5. **Training Process**
   - Plays random games to generate training data
   - Each game provides examples of board positions and outcomes
   - Backpropagation adjusts weights to improve predictions
   - More training = Better performance

#### **Key Formulas:**

**ReLU Activation (Hidden Layer):**
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

**Tanh Activation (Output Layer):**
```
f(x) = (e^x - e^-x) / (e^x + e^-x)
f'(x) = 1 - f(x)²
```

**Weight Update Rule:**
```
weight_new = weight_old - learning_rate × ∂Error/∂weight
```

#### **Strengths:**
- ✅ Can learn complex patterns from experience
- ✅ Adapts and improves with training
- ✅ Generalizes to new positions
- ✅ Fast evaluation (no tree search)

#### **Limitations:**
- ⚠️ Requires training to play well
- ⚠️ Quality depends on training data
- ⚠️ Less predictable than MiniMax
- ⚠️ May not play optimally without extensive training

---

### 📊 Comparison

| Feature | MiniMax | Neural Network |
|---------|---------|----------------|
| **Approach** | Rule-based search | Learning-based |
| **Training** | None required | Requires training |
| **Consistency** | Always optimal | Varies with training |
| **Speed** | Slower (tree search) | Faster (direct evaluation) |
| **Scalability** | Limited by depth | Scales better |
| **Explainability** | High (can trace moves) | Low (black box) |
| **Difficulty** | Very challenging | Moderate (if trained) |

### 🎯 Which Algorithm to Choose?

- **Choose MiniMax** if you want:
  - Maximum challenge
  - Consistent difficulty
  - Guaranteed optimal play

- **Choose Neural Network** if you want:
  - To see machine learning in action
  - Adaptive gameplay
  - Faster move calculations

## 📁 Project Structure

```
5x5TicTacTeo/
├── main_launcher.py              # Main entry point
├── gui_launcher.py               # GUI implementation
├── README_GUI.md                 # This file
├── MiniMax Algorithm/
│   └── FinalTicTacTeo.py        # MiniMax AI implementation
└── Backpropagation Algorithm/
    ├── game_logic.py            # Game logic and NN AI
    ├── neural_network.py        # Neural Network implementation
    ├── quick_start.py           # Quick start guide
    └── requirements.txt         # Python dependencies
```

## 🎓 Neural Network Training & Decision Making - Deep Dive

### 🧪 How to Train the Neural Network

The Neural Network learns by playing many games and adjusting its weights based on the outcomes.

#### **Method 1: Command Line Training**

```bash
cd "Backpropagation Algorithm"
python game_logic.py train 2000 300
```

**Parameters:**
- `2000` = Number of training games to simulate
- `300` = Number of epochs (complete passes through training data)

#### **Method 2: Custom Training Script**

Create a file `train_ai.py`:

```python
from game_logic import generate_training_data, train_neural_network
from neural_network import NeuralNetwork

# Step 1: Create a new neural network
nn = NeuralNetwork(
    input_size=25,      # 5x5 board = 25 cells
    hidden_size=50,     # 50 neurons in hidden layer
    output_size=1,      # 1 output (board evaluation)
    learning_rate=0.01  # How fast the network learns
)

# Step 2: Generate training data by playing random games
print("Generating training data...")
X_train, y_train = generate_training_data(num_games=2000)
# X_train: Board positions (2000 x 25 matrix)
# y_train: Outcomes (+1 win, -1 loss, 0 draw)

# Step 3: Train the network
print("Training neural network...")
train_neural_network(nn, X_train, y_train, epochs=300)

# Step 4: Save the trained model
nn.save('trained_model.pkl')
print("Training complete! Model saved.")
```

Run it:
```bash
python train_ai.py
```

---

### 🎯 How the Neural Network Makes Decisions

Let's walk through a complete example of how the AI decides which move to make.

#### **Step-by-Step Decision Process:**

**1. Current Game State**
```
Board:
  0 1 2 3 4
0 X - - - -
1 - O - - -
2 - - X - -
3 - - - - -
4 - - - O -

AI (O) needs to make a move
```

**2. Generate All Possible Moves**
```python
possible_moves = game_state.get_possible_moves()
# Result: [(0,1), (0,2), (0,3), ... (4,4)]  # 21 empty cells
```

**3. Evaluate Each Possible Move**

For each empty position, the AI:
1. Creates a hypothetical board with that move
2. Runs it through the neural network
3. Gets an evaluation score

```python
def get_ai_move_nn(game_state, nn):
    best_move = None
    best_score = -999999
    
    for move in possible_moves:
        # Simulate this move
        new_state = game_state.apply_move(move)
        
        # Check for immediate win
        if has_player_won(new_state, PLAYER_O):
            return move  # Take the winning move!
        
        # Evaluate position using neural network
        score = nn.evaluate_board(new_state.board)
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move
```

**4. Neural Network Evaluation (Forward Pass)**

For move at position (0, 1):

```python
# Step 4a: Convert board to input vector
board = [
    [-1,  0, 0, 0, 0],   # X=-1, O=1, Empty=0
    [ 0,  1, 0, 0, 0],
    [ 0,  0,-1, 0, 0],
    [ 0,  0, 0, 0, 0],
    [ 0,  0, 0, 1, 0]
]
input_vector = np.array(board).flatten()
# Result: [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, ... ]

# Step 4b: Hidden layer computation
hidden = np.dot(input_vector, weights1) + bias1
hidden = ReLU(hidden)  # Apply activation
# Result: [0.5, 2.3, 0, 1.8, ...]  # 50 values

# Step 4c: Output layer computation
output = np.dot(hidden, weights2) + bias2
score = tanh(output)  # Value between -1 and +1
# Result: 0.65  (Positive = good for AI)
```

**5. Compare All Moves**

```python
Move at (0,1): score = 0.65
Move at (0,2): score = 0.42
Move at (1,0): score = 0.73  ← Best move!
Move at (1,2): score = 0.38
Move at (2,0): score = 0.55
... (check all 21 possible moves)
```

**6. Select Best Move**

The AI chooses move (1, 0) with score 0.73

---

### 🔬 Training Process - What Actually Happens

#### **Data Generation Phase**

```python
def generate_training_data(num_games=100):
    X_train = []  # Store board positions
    y_train = []  # Store outcomes
    
    for game_num in range(num_games):
        # Play one complete game randomly
        board = create_empty_board()
        game_state = GameState(board, None, PLAYER_X)
        game_history = []
        
        # Play until game ends
        while not game_state.is_game_over():
            moves = game_state.get_possible_moves()
            move = random.choice(moves)  # Random move
            
            # Save this board position
            game_history.append(board.copy())
            
            game_state = game_state.apply_move(move)
        
        # Determine outcome
        if has_player_won(game_state, PLAYER_O):
            outcome = 1.0   # AI won
        elif has_player_won(game_state, PLAYER_X):
            outcome = -1.0  # AI lost
        else:
            outcome = 0.0   # Draw
        
        # Add all positions from this game with the outcome
        for board_position in game_history:
            X_train.append(flatten(board_position))
            y_train.append(outcome)
    
    return X_train, y_train
```

**Example Training Data:**
```
Board Position 1: [X, O, -, -, X, ...]  → Outcome: AI Won (+1)
Board Position 2: [-, X, O, -, -, ...]  → Outcome: AI Lost (-1)
Board Position 3: [O, -, X, X, O, ...]  → Outcome: Draw (0)
... (thousands more)
```

#### **Backpropagation Training Phase**

For each training example:

```python
def train_step(nn, board, target_outcome):
    # 1. Forward pass (prediction)
    predicted_score = nn.forward(board)
    
    # 2. Calculate error
    error = target_outcome - predicted_score
    # Example: If predicted 0.2 but outcome was 1.0
    # Error = 1.0 - 0.2 = 0.8 (prediction too low)
    
    # 3. Backward pass (calculate gradients)
    # Output layer gradient
    output_gradient = error * tanh_derivative(predicted_score)
    
    # Hidden layer gradient
    hidden_gradient = output_gradient * weights2 * relu_derivative(hidden)
    
    # 4. Update weights
    weights2 += learning_rate * hidden * output_gradient
    bias2 += learning_rate * output_gradient
    
    weights1 += learning_rate * input_vector * hidden_gradient
    bias1 += learning_rate * hidden_gradient
```

#### **Training Progress Example**

```
Epoch 1/300:
  Average Error: 0.85  (Network is guessing randomly)
  
Epoch 50/300:
  Average Error: 0.42  (Starting to learn patterns)
  
Epoch 150/300:
  Average Error: 0.18  (Getting better)
  
Epoch 300/300:
  Average Error: 0.08  (Much more accurate!)
```

---

### 🎮 Decision Making Example - Complete Walkthrough

**Scenario:** AI needs to block player from winning

```
Current Board:
  0 1 2 3 4
0 X X - - -    ← Player has 2 in a row!
1 - O - - -
2 - - - - -
3 - - - O -
4 - - - - -

AI's turn (O)
```

**AI Decision Process:**

```python
# 1. Get all possible moves
moves = [(0,2), (0,3), (0,4), (1,0), (1,2), ...]  # 21 moves

# 2. Evaluate each move
for move in moves:
    new_board = apply_move(current_board, move)
    
    # Special check: Does this move create a win for AI?
    if has_player_won(new_board, PLAYER_O):
        return move  # Take winning move immediately!
    
    # Special check: Does NOT taking this move let opponent win?
    if opponent_can_win_next(current_board, move):
        priority_moves.append(move)  # Block opponent!
    
    # Normal evaluation
    score = neural_network.evaluate(new_board)
    scores[move] = score

# 3. Check for critical moves
if (0,2) in priority_moves:  # This blocks opponent's win
    return (0,2)  # Block the threat!

# 4. Otherwise, choose best scored move
best_move = max(scores, key=scores.get)
return best_move
```

**Result:** AI plays at (0, 2) to block the player!

```
After AI Move:
  0 1 2 3 4
0 X X O - -    ← Blocked!
1 - O - - -
2 - - - - -
3 - - - O -
4 - - - - -
```

---

### 📊 Monitoring Training Quality

Check if your network is learning well:

```python
# Test the trained network
nn = NeuralNetwork.load('trained_model.pkl')

# Test position 1: Clear winning position for AI
test_board_1 = [
    [-1, -1,  0,  0,  0],
    [ 1,  1,  1,  0,  0],  # AI has 3 in a row
    [ 0,  0, -1,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0]
]
score_1 = nn.evaluate_board(test_board_1)
print(f"Winning position score: {score_1}")  
# Should be positive (e.g., 0.85)

# Test position 2: Losing position for AI
test_board_2 = [
    [ 1,  0,  0,  0,  0],
    [-1, -1, -1,  0,  0],  # Opponent has 3 in a row
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0]
]
score_2 = nn.evaluate_board(test_board_2)
print(f"Losing position score: {score_2}")
# Should be negative (e.g., -0.72)
```

**Good Training Indicators:**
- ✅ Winning positions get positive scores (> 0.5)
- ✅ Losing positions get negative scores (< -0.5)
- ✅ Neural network blocks obvious threats
- ✅ Neural network takes winning moves when available

---

### 🚀 Quick Training Commands

```bash
# Quick training (500 games, 100 epochs) - ~2 minutes
cd "Backpropagation Algorithm"
python game_logic.py train 500 100

# Medium training (2000 games, 300 epochs) - ~8 minutes
python game_logic.py train 2000 300

# Extensive training (5000 games, 500 epochs) - ~20 minutes
python game_logic.py train 5000 500
```

The more you train, the better the AI becomes!

## 🏆 Game Rules
- **Board**: 5x5 grid
- **Goal**: Get 3 of your symbols in a row
- **Directions**: Horizontal, Vertical, or Diagonal
- **Players**: You (X) vs AI (O)
- **Turn Order**: Player always goes first

## 🐛 Troubleshooting

### GUI doesn't start
- Make sure tkinter is installed (comes with most Python installations)
- On Linux: `sudo apt-get install python3-tk`

### Neural Network not working
- Ensure numpy is installed: `pip install numpy`
- If "No trained model found" message appears, the AI will use random initialization (train it for better performance)

### Import errors
- Make sure you're running the script from the correct directory
- Try: `cd "c:\Users\billa\OneDrive\Desktop\5x5TicTacTeo"`

## 📝 Notes
- The Neural Network AI may not play optimally if untrained
- Training the Neural Network improves its performance
- MiniMax AI provides consistent challenging gameplay
- Both AIs can be beaten with good strategy!

## 🎉 Enjoy the Game!
Have fun playing 5x5 Tic-Tac-Toe with AI opponents!
