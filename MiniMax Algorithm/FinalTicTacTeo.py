import copy
from typing import List, Tuple, Optional
import random

# Game constants
PLAYER_X = 'X'  # Human player
PLAYER_O = 'O'  # AI player
EMPTY_CELL = '-'
BOARD_SIZE = 5
WIN_LENGTH = 3
DEFAULT_SEARCH_DEPTH = 4

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

def get_score(game_state: GameState, player: str, depth: int) -> int:
    """Evaluates the game state score for the given player."""
    if has_player_won(game_state, player):
        return 100 + depth  # Favor faster wins
    elif has_player_won(game_state, get_next_player(player)):
        return -100 - depth  # Avoid faster losses
    else:
        return 0

def minimax(game_state: GameState, depth: int, alpha: float, beta: float, 
            maximizing_player: bool, ai_player: str) -> int:
    """Minimax algorithm with alpha-beta pruning."""
    if depth == 0 or game_state.is_game_over():
        return get_score(game_state, ai_player, depth)

    possible_moves = game_state.get_possible_moves()
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in possible_moves:
            new_state = game_state.apply_move(move)
            eval_score = minimax(new_state, depth - 1, alpha, beta, False, ai_player)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for move in possible_moves:
            new_state = game_state.apply_move(move)
            eval_score = minimax(new_state, depth - 1, alpha, beta, True, ai_player)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

def get_ai_move(game_state: GameState, depth: int = DEFAULT_SEARCH_DEPTH) -> Optional[Tuple[int, int]]:
    """Determines the best move for the AI using minimax algorithm."""
    possible_moves = game_state.get_possible_moves()
    if not possible_moves:
        return None

    best_move = possible_moves[0]
    best_score = float('-inf')
    
    for move in possible_moves:
        new_state = game_state.apply_move(move)
        # AI is maximizing player
        score = minimax(new_state, depth - 1, float('-inf'), float('inf'), False, PLAYER_O)
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move

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
    while True:
        # Initialize game state
        board = [[EMPTY_CELL] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        game_state = GameState(board, None, PLAYER_X)

        print("\n" + "="*40)
        print(f"Welcome to {BOARD_SIZE}x{BOARD_SIZE} Tic-Tac-Toe!")
        print(f"Get {WIN_LENGTH} in a row to win!")
        print("You are playing as 'X' against an AI opponent.")
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
                move = get_ai_move(game_state)
                print(f"AI chooses: Row {move[0]}, Column {move[1]}")

            game_state = game_state.apply_move(move)
            display_board(game_state.board)

            # Check for end of game
            if has_player_won(game_state, PLAYER_X):
                print("\n🎉 Congratulations! You won! 🎉")
                break
            elif has_player_won(game_state, PLAYER_O):
                print("\n💻 The AI won this time. Better luck next time!")
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
    play_game()

