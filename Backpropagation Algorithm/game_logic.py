import copy
import random
import numpy as np
from typing import List, Tuple, Optional
from neural_network import NeuralNetwork

# Game constants
PLAYER_X = 'X'  # Human player
PLAYER_O = 'O'  # AI player
EMPTY_CELL = '-'
BOARD_SIZE = 5
WIN_LENGTH = 3

class GameState:
    """Represents the current state of the game."""
    
    def __init__(self, board: List[List[str]], last_move: Optional[Tuple[int, int]], next_player: str):
        self.board = board
        self.last_move = last_move
        self.next_player = next_player

    def get_possible_moves(self) -> List[Tuple[int, int]]:
        """Returns list of all empty cells."""
        moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == EMPTY_CELL:
                    moves.append((i, j))
        return moves

    def is_valid_move(self, move: Tuple[int, int]) -> bool:
        """Checks if a move is valid."""
        row, col = move
        return (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and 
                self.board[row][col] == EMPTY_CELL)

    def apply_move(self, move: Tuple[int, int]) -> 'GameState':
        """Applies a move and returns a new game state."""
        new_board = copy.deepcopy(self.board)
        new_board[move[0]][move[1]] = self.next_player
        return GameState(new_board, move, get_next_player(self.next_player))
    
    def is_game_over(self) -> bool:
        """Checks if the game has ended."""
        if self.last_move is None:
            return False
        return (has_player_won(self, PLAYER_X) or 
                has_player_won(self, PLAYER_O) or 
                len(self.get_possible_moves()) == 0)

def get_next_player(player: str) -> str:
    """Returns the opposing player."""
    return PLAYER_O if player == PLAYER_X else PLAYER_X

def has_consecutive_symbols(line: List[str], player: str, length: int = WIN_LENGTH) -> bool:
    """Checks if a line contains the required consecutive symbols."""
    consecutive = 0
    for cell in line:
        if cell == player:
            consecutive += 1
            if consecutive >= length:
                return True
        else:
            consecutive = 0
    return False

def is_winning_row(row: List[str], player: str) -> bool:
    """Checks if a row contains a winning combination."""
    return has_consecutive_symbols(row, player)

def is_winning_column(board: List[List[str]], col: int, player: str) -> bool:
    """Checks if a column contains a winning combination."""
    column = [board[row][col] for row in range(BOARD_SIZE)]
    return has_consecutive_symbols(column, player)

def is_winning_diagonal(board: List[List[str]], player: str) -> bool:
    """Checks all possible diagonals for a winning combination."""
    # Check all diagonals (top-left to bottom-right)
    for start_row in range(BOARD_SIZE - WIN_LENGTH + 1):
        for start_col in range(BOARD_SIZE - WIN_LENGTH + 1):
            diagonal = [board[start_row + i][start_col + i] for i in range(BOARD_SIZE - max(start_row, start_col))]
            if has_consecutive_symbols(diagonal, player):
                return True
    
    # Check all anti-diagonals (top-right to bottom-left)
    for start_row in range(BOARD_SIZE - WIN_LENGTH + 1):
        for start_col in range(WIN_LENGTH - 1, BOARD_SIZE):
            diagonal = [board[start_row + i][start_col - i] for i in range(min(BOARD_SIZE - start_row, start_col + 1))]
            if has_consecutive_symbols(diagonal, player):
                return True
    
    return False

def has_player_won(game_state: GameState, player: str) -> bool:
    """Checks if the specified player has won the game."""
    if game_state.last_move is None:
        return False
    
    board = game_state.board
    last_row, last_col = game_state.last_move
    
    # Check row, column, and diagonals
    return (is_winning_row(board[last_row], player) or
            is_winning_column(board, last_col, player) or
            is_winning_diagonal(board, player))

# Global neural network instance
neural_network = None

def initialize_ai(model_path: str = "trained_model.pkl"):
    """Initialize the AI neural network."""
    global neural_network
    neural_network = NeuralNetwork(input_size=25, hidden_size=50, output_size=1, learning_rate=0.01)
    
    # Try to load pre-trained model
    if neural_network.load(model_path):
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print("No pre-trained model found. Using random initialization.")
        print("The AI will improve as it plays more games!")

def evaluate_position(game_state: GameState) -> float:
    """
    Evaluate a game position using the neural network.
    Positive score favors AI (O), negative favors player (X).
    """
    if neural_network is None:
        initialize_ai()
    
    # Check terminal states first
    if has_player_won(game_state, PLAYER_O):
        return 1.0
    elif has_player_won(game_state, PLAYER_X):
        return -1.0
    elif not game_state.get_possible_moves():
        return 0.0
    
    # Use neural network to evaluate
    return neural_network.evaluate_board(game_state.board)

def get_ai_move(game_state: GameState, exploration_rate: float = 0.1) -> Optional[Tuple[int, int]]:
    """
    Determines the best move for the AI using neural network evaluation.
    
    Args:
        game_state: Current game state
        exploration_rate: Probability of making a random move (for learning)
    
    Returns:
        Best move as (row, col) tuple
    """
    possible_moves = game_state.get_possible_moves()
    if not possible_moves:
        return None
    
    # Epsilon-greedy strategy: explore with small probability
    if random.random() < exploration_rate:
        return random.choice(possible_moves)
    
    # Evaluate all possible moves
    best_move = possible_moves[0]
    best_score = float('-inf')
    
    for move in possible_moves:
        new_state = game_state.apply_move(move)
        
        # Check for immediate win
        if has_player_won(new_state, PLAYER_O):
            return move
        
        # Evaluate position
        score = evaluate_position(new_state)
        
        if score > best_score:
            best_score = score
            best_move = move
    
    # Check if opponent can win next move and block it
    for move in possible_moves:
        test_board = copy.deepcopy(game_state.board)
        test_board[move[0]][move[1]] = PLAYER_X
        test_state = GameState(test_board, move, PLAYER_O)
        if has_player_won(test_state, PLAYER_X):
            return move  # Block the winning move
    
    return best_move

def get_ai_move_nn(game_state: GameState, nn: NeuralNetwork, exploration_rate: float = 0.0) -> Optional[Tuple[int, int]]:
    """
    Determines the best move for the AI using a provided neural network.
    This version accepts a neural network instance as parameter.
    
    Args:
        game_state: Current game state
        nn: Neural network instance to use for evaluation
        exploration_rate: Probability of making a random move
    
    Returns:
        Best move as (row, col) tuple
    """
    possible_moves = game_state.get_possible_moves()
    if not possible_moves:
        return None
    
    # Epsilon-greedy strategy
    if random.random() < exploration_rate:
        return random.choice(possible_moves)
    
    # Evaluate all possible moves
    best_move = possible_moves[0]
    best_score = float('-inf')
    
    for move in possible_moves:
        new_state = game_state.apply_move(move)
        
        # Check for immediate win
        if has_player_won(new_state, PLAYER_O):
            return move
        
        # Evaluate position using provided neural network
        if has_player_won(new_state, PLAYER_O):
            score = 1.0
        elif has_player_won(new_state, PLAYER_X):
            score = -1.0
        elif not new_state.get_possible_moves():
            score = 0.0
        else:
            score = nn.evaluate_board(new_state.board)
        
        if score > best_score:
            best_score = score
            best_move = move
    
    # Check if opponent can win next move and block it
    for move in possible_moves:
        test_board = copy.deepcopy(game_state.board)
        test_board[move[0]][move[1]] = PLAYER_X
        test_state = GameState(test_board, move, PLAYER_O)
        if has_player_won(test_state, PLAYER_X):
            return move  # Block the winning move
    
    return best_move

def generate_training_data(num_games: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data by playing random games.
    
    Args:
        num_games: Number of games to simulate
    
    Returns:
        Tuple of (training inputs, training outputs)
    """
    X_train = []
    y_train = []
    
    for game_num in range(num_games):
        board = [[EMPTY_CELL] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        game_state = GameState(board, None, PLAYER_X)
        game_history = []
        
        # Play a random game
        while not game_state.is_game_over():
            moves = game_state.get_possible_moves()
            if not moves:
                break
            
            move = random.choice(moves)
            game_history.append((copy.deepcopy(game_state.board), game_state.next_player))
            game_state = game_state.apply_move(move)
        
        # Determine outcome
        if has_player_won(game_state, PLAYER_O):
            outcome = 1.0  # AI won
        elif has_player_won(game_state, PLAYER_X):
            outcome = -1.0  # Player won
        else:
            outcome = 0.0  # Draw
        
        # Create training samples from game history
        for board_state, player in game_history:
            if neural_network is None:
                initialize_ai()
            
            input_vector = neural_network.board_to_vector(board_state)
            X_train.append(input_vector)
            
            # Assign value based on outcome and whose turn it was
            if player == PLAYER_O:
                y_train.append([outcome])
            else:
                y_train.append([-outcome])
    
    return np.array(X_train), np.array(y_train)

def train_neural_network(num_games: int = 500, epochs: int = 100, save_path: str = "trained_model.pkl"):
    """
    Train the neural network on random game data.
    
    Args:
        num_games: Number of games to generate for training
        epochs: Number of training epochs
        save_path: Path to save the trained model
    """
    print(f"Generating training data from {num_games} random games...")
    X_train, y_train = generate_training_data(num_games)
    
    print(f"Training neural network for {epochs} epochs...")
    if neural_network is None:
        initialize_ai()
    
    neural_network.train(X_train, y_train, epochs=epochs, verbose=True)
    
    print(f"Saving trained model to {save_path}...")
    neural_network.save(save_path)
    print("Training complete!")

def display_board(board: List[List[str]]) -> None:
    """Displays the game board with coordinates."""
    print("\n  " + " ".join(str(i) for i in range(BOARD_SIZE)))
    print("  " + "-" * (BOARD_SIZE * 2 - 1))
    for i in range(BOARD_SIZE):
        row_str = str(i) + "|" + " ".join(board[i])
        print(row_str)
    print()

def get_user_move(game_state: GameState) -> Tuple[int, int]:
    """Gets and validates user input for their move."""
    while True:
        try:
            row = int(input(f"Enter row number (0-{BOARD_SIZE-1}): "))
            col = int(input(f"Enter column number (0-{BOARD_SIZE-1}): "))
            move = (row, col)
            
            if not game_state.is_valid_move(move):
                print("Invalid move! Cell is either out of bounds or already occupied.")
                continue
            
            return move
        except ValueError:
            print(f"Invalid input. Please enter a number from 0 to {BOARD_SIZE-1}.")

def play_game() -> None:
    """Main game loop."""
    # Initialize AI
    initialize_ai()
    
    while True:
        # Initialize game state
        board = [[EMPTY_CELL] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        game_state = GameState(board, None, PLAYER_X)

        print("\n" + "="*40)
        print(f"Welcome to {BOARD_SIZE}x{BOARD_SIZE} Tic-Tac-Toe!")
        print(f"Get {WIN_LENGTH} in a row to win!")
        print("You are playing as 'X' against a Neural Network AI.")
        print("="*40)
        display_board(game_state.board)

        move_count = 0
        while True:
            move_count += 1
            
            # Player X's turn (user)
            if game_state.next_player == PLAYER_X:
                print(f"\n[Move {move_count}] Your turn!")
                move = get_user_move(game_state)
            # Player O's turn (AI)
            else:
                print(f"\n[Move {move_count}] AI is thinking...")
                move = get_ai_move(game_state, exploration_rate=0.0)  # No exploration during play
                print(f"AI chooses: Row {move[0]}, Column {move[1]}")

            game_state = game_state.apply_move(move)
            display_board(game_state.board)

            # Check for end of game
            if has_player_won(game_state, PLAYER_X):
                print("\n🎉 Congratulations! You won! 🎉")
                break
            elif has_player_won(game_state, PLAYER_O):
                print("\n💻 The Neural Network AI won this time!")
                break
            elif not game_state.get_possible_moves():
                print("\n🤝 It's a tie! Well played!")
                break
        
        # Ask for replay
        while True:
            replay = input("\nWould you like to play again? (yes/no): ").lower().strip()
            if replay in ['yes', 'y']:
                break
            elif replay in ['no', 'n']:
                print("\nThanks for playing! Goodbye!")
                return
            else:
                print("Please enter 'yes' or 'no'.")

if __name__ == "__main__":
    import sys
    
    # Check if training mode
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        num_games = 1000 if len(sys.argv) < 3 else int(sys.argv[2])
        epochs = 200 if len(sys.argv) < 4 else int(sys.argv[3])
        train_neural_network(num_games=num_games, epochs=epochs)
    else:
        play_game()
