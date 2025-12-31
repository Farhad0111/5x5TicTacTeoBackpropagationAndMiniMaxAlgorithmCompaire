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

## 🎓 Training the Neural Network
If you want to improve the Neural Network AI:

```bash
cd "Backpropagation Algorithm"
python game_logic.py train 2000 300
```

This will train the AI with 2000 games and 300 epochs.

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
#   5 x 5 T i c T a c T e o B a c k p r o p a g a t i o n A n d M i n i M a x A l g o r i t h m C o m p a i r e  
 