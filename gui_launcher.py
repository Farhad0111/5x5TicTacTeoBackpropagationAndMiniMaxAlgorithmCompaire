"""
5x5 Tic-Tac-Toe Game Launcher with GUI
=======================================
Main launcher with menu to select between MiniMax and Neural Network AI
"""

import tkinter as tk
from tkinter import messagebox, font
import sys
import os

# Add the algorithm directories to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'MiniMax Algorithm'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Backpropagation Algorithm'))

class MenuGUI:
    """Main menu GUI for game selection"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("5x5 Tic-Tac-Toe - Game Selection")
        self.window.geometry("600x500")
        self.window.configure(bg='#2C3E50')
        
        # Custom fonts
        self.title_font = font.Font(family='Arial', size=24, weight='bold')
        self.button_font = font.Font(family='Arial', size=14, weight='bold')
        self.desc_font = font.Font(family='Arial', size=10)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the main menu UI"""
        
        # Title
        title_frame = tk.Frame(self.window, bg='#34495E', pady=20)
        title_frame.pack(fill='x', pady=(0, 30))
        
        title_label = tk.Label(
            title_frame,
            text="🎮 5x5 Tic-Tac-Toe 🎮",
            font=self.title_font,
            bg='#34495E',
            fg='#ECF0F1'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Select Your Game Mode",
            font=('Arial', 12),
            bg='#34495E',
            fg='#BDC3C7'
        )
        subtitle_label.pack()
        
        # Main content frame
        content_frame = tk.Frame(self.window, bg='#2C3E50')
        content_frame.pack(expand=True, fill='both', padx=40)
        
        # MiniMax Algorithm Option
        minimax_frame = tk.Frame(content_frame, bg='#1ABC9C', relief='raised', borderwidth=3)
        minimax_frame.pack(fill='x', pady=10)
        
        minimax_title = tk.Label(
            minimax_frame,
            text="🤖 MiniMax Algorithm",
            font=self.button_font,
            bg='#1ABC9C',
            fg='white',
            pady=10
        )
        minimax_title.pack()
        
        minimax_desc = tk.Label(
            minimax_frame,
            text="Classic AI using MiniMax algorithm with Alpha-Beta pruning\n"
                 "Perfect strategic play - Challenging opponent!",
            font=self.desc_font,
            bg='#1ABC9C',
            fg='white',
            justify='center',
            pady=5
        )
        minimax_desc.pack()
        
        minimax_btn = tk.Button(
            minimax_frame,
            text="Play with MiniMax AI",
            font=('Arial', 12, 'bold'),
            bg='#16A085',
            fg='white',
            activebackground='#138D75',
            activeforeground='white',
            command=self.launch_minimax_game,
            cursor='hand2',
            pady=10,
            relief='raised',
            borderwidth=2
        )
        minimax_btn.pack(fill='x', padx=20, pady=10)
        
        # Neural Network Option
        nn_frame = tk.Frame(content_frame, bg='#3498DB', relief='raised', borderwidth=3)
        nn_frame.pack(fill='x', pady=10)
        
        nn_title = tk.Label(
            nn_frame,
            text="🧠 Neural Network (Backpropagation)",
            font=self.button_font,
            bg='#3498DB',
            fg='white',
            pady=10
        )
        nn_title.pack()
        
        nn_desc = tk.Label(
            nn_frame,
            text="AI trained with Neural Network using Backpropagation\n"
                 "Learning-based approach - Adaptive gameplay!",
            font=self.desc_font,
            bg='#3498DB',
            fg='white',
            justify='center',
            pady=5
        )
        nn_desc.pack()
        
        nn_btn = tk.Button(
            nn_frame,
            text="Play with Neural Network AI",
            font=('Arial', 12, 'bold'),
            bg='#2980B9',
            fg='white',
            activebackground='#21618C',
            activeforeground='white',
            command=self.launch_nn_game,
            cursor='hand2',
            pady=10,
            relief='raised',
            borderwidth=2
        )
        nn_btn.pack(fill='x', padx=20, pady=10)
        
        # Footer
        footer_frame = tk.Frame(self.window, bg='#2C3E50')
        footer_frame.pack(side='bottom', fill='x', pady=10)
        
        info_label = tk.Label(
            footer_frame,
            text="Goal: Get 3 in a row to win! (Horizontal, Vertical, or Diagonal)",
            font=('Arial', 9),
            bg='#2C3E50',
            fg='#95A5A6'
        )
        info_label.pack()
        
        exit_btn = tk.Button(
            footer_frame,
            text="Exit",
            font=('Arial', 10),
            bg='#E74C3C',
            fg='white',
            activebackground='#C0392B',
            command=self.window.quit,
            cursor='hand2',
            padx=20,
            pady=5
        )
        exit_btn.pack(pady=5)
        
    def launch_minimax_game(self):
        """Launch the MiniMax algorithm game"""
        self.window.withdraw()
        try:
            from FinalTicTacTeo import GameState, play_game, PLAYER_X, PLAYER_O, EMPTY_CELL, BOARD_SIZE
            game_window = MinimaxGameGUI(self.window)
            game_window.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch MiniMax game:\n{str(e)}")
            self.window.deiconify()
    
    def launch_nn_game(self):
        """Launch the Neural Network game"""
        self.window.withdraw()
        try:
            from game_logic import GameState, PLAYER_X, PLAYER_O, EMPTY_CELL, BOARD_SIZE
            from neural_network import NeuralNetwork
            game_window = NeuralNetworkGameGUI(self.window)
            game_window.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Neural Network game:\n{str(e)}")
            self.window.deiconify()
    
    def run(self):
        """Start the main menu"""
        self.window.mainloop()


class MinimaxGameGUI:
    """GUI for MiniMax Algorithm Game"""
    
    def __init__(self, parent):
        from FinalTicTacTeo import GameState, PLAYER_X, PLAYER_O, EMPTY_CELL, BOARD_SIZE, get_ai_move, has_player_won
        
        self.GameState = GameState
        self.PLAYER_X = PLAYER_X
        self.PLAYER_O = PLAYER_O
        self.EMPTY_CELL = EMPTY_CELL
        self.BOARD_SIZE = BOARD_SIZE
        self.get_ai_move = get_ai_move
        self.has_player_won = has_player_won
        
        self.parent = parent
        self.window = tk.Toplevel()
        self.window.title("5x5 Tic-Tac-Toe - MiniMax AI")
        self.window.geometry("600x700")
        self.window.configure(bg='#2C3E50')
        
        self.buttons = []
        self.game_state = None
        self.game_over = False
        
        self.setup_ui()
        self.new_game()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        """Set up the game UI"""
        
        # Title
        title_frame = tk.Frame(self.window, bg='#1ABC9C', pady=15)
        title_frame.pack(fill='x')
        
        tk.Label(
            title_frame,
            text="🤖 Playing against MiniMax AI",
            font=('Arial', 18, 'bold'),
            bg='#1ABC9C',
            fg='white'
        ).pack()
        
        # Status label
        self.status_label = tk.Label(
            self.window,
            text="Your turn! (You are X)",
            font=('Arial', 14),
            bg='#2C3E50',
            fg='#ECF0F1',
            pady=10
        )
        self.status_label.pack()
        
        # Game board
        board_frame = tk.Frame(self.window, bg='#34495E')
        board_frame.pack(pady=20)
        
        for i in range(self.BOARD_SIZE):
            row = []
            for j in range(self.BOARD_SIZE):
                btn = tk.Button(
                    board_frame,
                    text='',
                    font=('Arial', 20, 'bold'),
                    width=4,
                    height=2,
                    bg='#ECF0F1',
                    fg='#2C3E50',
                    activebackground='#BDC3C7',
                    command=lambda r=i, c=j: self.make_move(r, c),
                    relief='raised',
                    borderwidth=3
                )
                btn.grid(row=i, column=j, padx=2, pady=2)
                row.append(btn)
            self.buttons.append(row)
        
        # Control buttons
        control_frame = tk.Frame(self.window, bg='#2C3E50')
        control_frame.pack(pady=20)
        
        tk.Button(
            control_frame,
            text="New Game",
            font=('Arial', 12, 'bold'),
            bg='#3498DB',
            fg='white',
            activebackground='#2980B9',
            command=self.new_game,
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(side='left', padx=5)
        
        tk.Button(
            control_frame,
            text="Back to Menu",
            font=('Arial', 12, 'bold'),
            bg='#E74C3C',
            fg='white',
            activebackground='#C0392B',
            command=self.on_closing,
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(side='left', padx=5)
    
    def new_game(self):
        """Start a new game"""
        board = [[self.EMPTY_CELL for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.game_state = self.GameState(board, None, self.PLAYER_X)
        self.game_over = False
        self.update_board()
        self.status_label.config(text="Your turn! (You are X)", fg='#ECF0F1')
    
    def update_board(self):
        """Update the button display"""
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                cell = self.game_state.board[i][j]
                self.buttons[i][j].config(text=cell if cell != self.EMPTY_CELL else '')
                
                # Color coding
                if cell == self.PLAYER_X:
                    self.buttons[i][j].config(fg='#E74C3C')  # Red for X
                elif cell == self.PLAYER_O:
                    self.buttons[i][j].config(fg='#3498DB')  # Blue for O
                else:
                    self.buttons[i][j].config(fg='#2C3E50')
    
    def make_move(self, row, col):
        """Handle player move"""
        if self.game_over:
            return
        
        if not self.game_state.is_valid_move((row, col)):
            return
        
        # Player move
        self.game_state = self.game_state.apply_move((row, col))
        self.update_board()
        
        # Check for win
        if self.check_game_over():
            return
        
        # AI move
        self.status_label.config(text="AI is thinking...", fg='#F39C12')
        self.window.update()
        self.window.after(500, self.ai_move)
    
    def ai_move(self):
        """Execute AI move"""
        if self.game_over:
            return
        
        move = self.get_ai_move(self.game_state)
        self.game_state = self.game_state.apply_move(move)
        self.update_board()
        
        self.check_game_over()
    
    def check_game_over(self):
        """Check if game is over"""
        if self.has_player_won(self.game_state, self.PLAYER_X):
            self.status_label.config(text="🎉 You Won! 🎉", fg='#2ECC71')
            self.game_over = True
            messagebox.showinfo("Game Over", "Congratulations! You won!")
            return True
        elif self.has_player_won(self.game_state, self.PLAYER_O):
            self.status_label.config(text="💻 AI Won!", fg='#E74C3C')
            self.game_over = True
            messagebox.showinfo("Game Over", "The AI won this time!")
            return True
        elif not self.game_state.get_possible_moves():
            self.status_label.config(text="🤝 It's a Tie!", fg='#F39C12')
            self.game_over = True
            messagebox.showinfo("Game Over", "It's a tie!")
            return True
        else:
            self.status_label.config(text="Your turn! (You are X)", fg='#ECF0F1')
            return False
    
    def on_closing(self):
        """Handle window close"""
        self.window.destroy()
        self.parent.deiconify()
    
    def run(self):
        """Run the game window"""
        pass


class NeuralNetworkGameGUI:
    """GUI for Neural Network Game"""
    
    def __init__(self, parent):
        from game_logic import GameState, PLAYER_X, PLAYER_O, EMPTY_CELL, BOARD_SIZE, has_player_won, get_ai_move_nn
        from neural_network import NeuralNetwork
        
        self.GameState = GameState
        self.PLAYER_X = PLAYER_X
        self.PLAYER_O = PLAYER_O
        self.EMPTY_CELL = EMPTY_CELL
        self.BOARD_SIZE = BOARD_SIZE
        self.has_player_won = has_player_won
        self.get_ai_move_nn = get_ai_move_nn
        self.NeuralNetwork = NeuralNetwork
        
        self.parent = parent
        self.window = tk.Toplevel()
        self.window.title("5x5 Tic-Tac-Toe - Neural Network AI")
        self.window.geometry("600x700")
        self.window.configure(bg='#2C3E50')
        
        self.buttons = []
        self.game_state = None
        self.game_over = False
        self.nn = None
        
        self.load_neural_network()
        self.setup_ui()
        self.new_game()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def load_neural_network(self):
        """Load or create neural network"""
        self.nn = self.NeuralNetwork()
        try:
            if self.nn.load('trained_model.pkl'):
                pass  # Model loaded successfully
            elif self.nn.load('Backpropagation Algorithm/trained_model.pkl'):
                pass  # Model loaded from subdirectory
            else:
                messagebox.showinfo("Neural Network", 
                                       "No trained model found. Using untrained network.\n"
                                       "The AI might not play optimally.")
        except Exception as e:
            messagebox.showinfo("Neural Network", 
                                   f"Using untrained network.\n"
                                   f"The AI might not play optimally.")
    
    def setup_ui(self):
        """Set up the game UI"""
        
        # Title
        title_frame = tk.Frame(self.window, bg='#3498DB', pady=15)
        title_frame.pack(fill='x')
        
        tk.Label(
            title_frame,
            text="🧠 Playing against Neural Network AI",
            font=('Arial', 18, 'bold'),
            bg='#3498DB',
            fg='white'
        ).pack()
        
        # Status label
        self.status_label = tk.Label(
            self.window,
            text="Your turn! (You are X)",
            font=('Arial', 14),
            bg='#2C3E50',
            fg='#ECF0F1',
            pady=10
        )
        self.status_label.pack()
        
        # Game board
        board_frame = tk.Frame(self.window, bg='#34495E')
        board_frame.pack(pady=20)
        
        for i in range(self.BOARD_SIZE):
            row = []
            for j in range(self.BOARD_SIZE):
                btn = tk.Button(
                    board_frame,
                    text='',
                    font=('Arial', 20, 'bold'),
                    width=4,
                    height=2,
                    bg='#ECF0F1',
                    fg='#2C3E50',
                    activebackground='#BDC3C7',
                    command=lambda r=i, c=j: self.make_move(r, c),
                    relief='raised',
                    borderwidth=3
                )
                btn.grid(row=i, column=j, padx=2, pady=2)
                row.append(btn)
            self.buttons.append(row)
        
        # Control buttons
        control_frame = tk.Frame(self.window, bg='#2C3E50')
        control_frame.pack(pady=20)
        
        tk.Button(
            control_frame,
            text="New Game",
            font=('Arial', 12, 'bold'),
            bg='#3498DB',
            fg='white',
            activebackground='#2980B9',
            command=self.new_game,
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(side='left', padx=5)
        
        tk.Button(
            control_frame,
            text="Back to Menu",
            font=('Arial', 12, 'bold'),
            bg='#E74C3C',
            fg='white',
            activebackground='#C0392B',
            command=self.on_closing,
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(side='left', padx=5)
    
    def new_game(self):
        """Start a new game"""
        board = [[self.EMPTY_CELL for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.game_state = self.GameState(board, None, self.PLAYER_X)
        self.game_over = False
        self.update_board()
        self.status_label.config(text="Your turn! (You are X)", fg='#ECF0F1')
    
    def update_board(self):
        """Update the button display"""
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                cell = self.game_state.board[i][j]
                self.buttons[i][j].config(text=cell if cell != self.EMPTY_CELL else '')
                
                # Color coding
                if cell == self.PLAYER_X:
                    self.buttons[i][j].config(fg='#E74C3C')  # Red for X
                elif cell == self.PLAYER_O:
                    self.buttons[i][j].config(fg='#3498DB')  # Blue for O
                else:
                    self.buttons[i][j].config(fg='#2C3E50')
    
    def make_move(self, row, col):
        """Handle player move"""
        if self.game_over:
            return
        
        if not self.game_state.is_valid_move((row, col)):
            return
        
        # Player move
        self.game_state = self.game_state.apply_move((row, col))
        self.update_board()
        
        # Check for win
        if self.check_game_over():
            return
        
        # AI move
        self.status_label.config(text="AI is thinking...", fg='#F39C12')
        self.window.update()
        self.window.after(500, self.ai_move)
    
    def ai_move(self):
        """Execute AI move"""
        if self.game_over:
            return
        
        move = self.get_ai_move_nn(self.game_state, self.nn)
        self.game_state = self.game_state.apply_move(move)
        self.update_board()
        
        self.check_game_over()
    
    def check_game_over(self):
        """Check if game is over"""
        if self.has_player_won(self.game_state, self.PLAYER_X):
            self.status_label.config(text="🎉 You Won! 🎉", fg='#2ECC71')
            self.game_over = True
            messagebox.showinfo("Game Over", "Congratulations! You won!")
            return True
        elif self.has_player_won(self.game_state, self.PLAYER_O):
            self.status_label.config(text="💻 AI Won!", fg='#E74C3C')
            self.game_over = True
            messagebox.showinfo("Game Over", "The AI won this time!")
            return True
        elif not self.game_state.get_possible_moves():
            self.status_label.config(text="🤝 It's a Tie!", fg='#F39C12')
            self.game_over = True
            messagebox.showinfo("Game Over", "It's a tie!")
            return True
        else:
            self.status_label.config(text="Your turn! (You are X)", fg='#ECF0F1')
            return False
    
    def on_closing(self):
        """Handle window close"""
        self.window.destroy()
        self.parent.deiconify()
    
    def run(self):
        """Run the game window"""
        pass


if __name__ == "__main__":
    app = MenuGUI()
    app.run()
