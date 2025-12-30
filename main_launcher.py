"""
5x5 Tic-Tac-Toe Game Launcher
==============================
Main entry point for the 5x5 Tic-Tac-Toe game with GUI

Simply run this file to start the game:
    python main_launcher.py

Features:
- Beautiful GUI menu to select game mode
- MiniMax Algorithm AI (Classic strategic AI)
- Neural Network AI (Learning-based AI with Backpropagation)
- Play against either AI in an intuitive interface
"""

import subprocess
import sys
import os

def main():
    """Launch the GUI game launcher"""
    launcher_path = os.path.join(os.path.dirname(__file__), 'gui_launcher.py')
    
    try:
        # Run the GUI launcher
        subprocess.run([sys.executable, launcher_path])
    except Exception as e:
        print(f"Error launching game: {e}")
        print("\nTrying alternative method...")
        # Alternative: import and run directly
        try:
            import gui_launcher
            app = gui_launcher.MenuGUI()
            app.run()
        except Exception as e2:
            print(f"Error: {e2}")
            print("\nPlease ensure all required dependencies are installed:")
            print("  - tkinter (usually included with Python)")
            print("  - numpy")

if __name__ == "__main__":
    print("=" * 60)
    print(" 🎮 5x5 Tic-Tac-Toe Game Launcher 🎮")
    print("=" * 60)
    print("\nStarting game...")
    main()
