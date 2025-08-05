"""
Tic-Tac-Toe Game Service

This module provides a self-contained Tic-Tac-Toe game service designed for human vs. AI gameplay.
It implements a minimax algorithm for optimal AI moves, making the AI unbeatable.
The service manages game state, player turns, and outcome detection autonomously.

Usage:
    game = TicTacToeGame()  # Initialize with default symbols ('X' for human, 'O' for AI)
    
    # Human makes a move at index 4 (center cell)
    result = game.make_move(4)
    
    # AI responds automatically
    ai_result = game.ai_move()
    
    # Get the current game state
    state = game.get_game_state()
"""

import copy
import json
import random

class TicTacToeGame:
    """
    A self-contained Tic-Tac-Toe game class that manages the game state and provides
    methods for gameplay, including an unbeatable AI opponent using the minimax algorithm.
    """
    
    def __init__(self, human_symbol='X', ai_symbol='O'):
        """
        Initialize a new Tic-Tac-Toe game with customizable player symbols.
        
        Args:
            human_symbol (str): Symbol for the human player (default: 'X')
            ai_symbol (str): Symbol for the AI player (default: 'O')
        
        Raises:
            ValueError: If symbols are identical or not single characters
        """
        # Validate symbols
        if human_symbol == ai_symbol:
            raise ValueError("Human and AI symbols must be different")
        if len(human_symbol) != 1 or len(ai_symbol) != 1:
            raise ValueError("Symbols must be single characters")
            
        self.human_symbol = human_symbol
        self.ai_symbol = ai_symbol
        
        # Initialize game state
        self.board = [None] * 9  # 3x3 board represented as 1D array
        self.current_player = human_symbol  # Human goes first
        self.game_over = False
        self.winner = None
        
        # Win conditions (indices of winning combinations)
        self.win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]

    def reset_game(self):
        """
        Reset the game to its initial state.
        
        Returns:
            dict: The initial game state
        """
        self.board = [None] * 9
        self.current_player = self.human_symbol
        self.game_over = False
        self.winner = None
        return self.serialize_game_state()
    
    def check_win(self, board, player):
        """
        Check if the specified player has won on the given board.
        
        Args:
            board (list): The game board to check
            player (str): The player symbol to check for a win
            
        Returns:
            bool: True if the player has won, False otherwise
        """
        for condition in self.win_conditions:
            if all(board[i] == player for i in condition):
                return True
        return False
    
    def check_draw(self, board):
        """
        Check if the game is a draw (board full with no winner).
        
        Args:
            board (list): The game board to check
            
        Returns:
            bool: True if the game is a draw, False otherwise
        """
        return None not in board and not self.check_win(board, self.human_symbol) and not self.check_win(board, self.ai_symbol)
    
    def get_available_moves(self, board):
        """
        Get all available (empty) cell indices on the board.
        
        Args:
            board (list): The game board to check
            
        Returns:
            list: Indices of empty cells
        """
        return [i for i, cell in enumerate(board) if cell is None]
    
    def get_game_state(self):
        """
        Get the current game state.
        
        Returns:
            dict: Game state with status and winner (if applicable)
        """
        if self.check_win(self.board, self.human_symbol):
            return {'status': 'win', 'winner': self.human_symbol}
        elif self.check_win(self.board, self.ai_symbol):
            return {'status': 'win', 'winner': self.ai_symbol}
        elif self.check_draw(self.board):
            return {'status': 'draw'}
        else:
            return {'status': 'ongoing'}
    
    def make_move(self, index):
        """
        Process a human move at the specified index.
        
        Args:
            index (int): Board position (0-8) for the move
            
        Returns:
            dict: Updated game state after the move
            
        Raises:
            ValueError: If the move is invalid or the game is over
        """
        # Validate move
        if self.game_over:
            raise ValueError("Game is already over")
        
        if not (0 <= index <= 8):
            raise ValueError(f"Invalid index: {index}. Must be between 0 and 8.")
            
        if self.board[index] is not None:
            raise ValueError(f"Cell {index} is already occupied")
            
        if self.current_player != self.human_symbol:
            raise ValueError("It's not your turn")
            
        # Make the move
        self.board[index] = self.human_symbol
        
        # Update game state
        game_state = self.get_game_state()
        if game_state['status'] != 'ongoing':
            self.game_over = True
            if game_state['status'] == 'win':
                self.winner = game_state['winner']
        else:
            self.current_player = self.ai_symbol
            
        return self.serialize_game_state()
    
    def minimax(self, board, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
        """
        Minimax algorithm with alpha-beta pruning to find the optimal move.
        
        Args:
            board (list): Current board state
            depth (int): Current depth in the game tree
            is_maximizing (bool): True if maximizing player's turn, False otherwise
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            
        Returns:
            tuple: (score, move) pair representing the best score and corresponding move
        """
        # Base cases: check terminal states
        if self.check_win(board, self.ai_symbol):
            return 10 - depth, None
        elif self.check_win(board, self.human_symbol):
            return depth - 10, None
        elif self.check_draw(board):
            return 0, None
            
        # Recursive case
        available_moves = self.get_available_moves(board)
        
        if is_maximizing:  # AI's turn (maximizing)
            best_score = -float('inf')
            best_move = None
            
            for move in available_moves:
                new_board = board.copy()
                new_board[move] = self.ai_symbol
                
                score, _ = self.minimax(new_board, depth + 1, False, alpha, beta)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
                    
            return best_score, best_move
        else:  # Human's turn (minimizing)
            best_score = float('inf')
            best_move = None
            
            for move in available_moves:
                new_board = board.copy()
                new_board[move] = self.human_symbol
                
                score, _ = self.minimax(new_board, depth + 1, True, alpha, beta)
                
                if score < best_score:
                    best_score = score
                    best_move = move
                    
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
                    
            return best_score, best_move
    
    def get_best_move(self, board):
        """
        Get the best move for the AI using the minimax algorithm.
        
        Args:
            board (list): Current board state
            
        Returns:
            int: Index of the best move
        """
        # Add a bit of randomness for the first move to make the game more interesting
        available_moves = self.get_available_moves(board)
        if len(available_moves) == 9:  # First move
            return random.choice([0, 2, 4, 6, 8])  # Choose from corners and center
        
        # For subsequent moves, use minimax
        _, best_move = self.minimax(board, 0, True)
        return best_move
    
    def ai_move(self):
        """
        Calculate and execute the AI's optimal move.
        
        Returns:
            tuple: (move, game_state) - the move made and updated game state
            
        Raises:
            ValueError: If the game is over or it's not AI's turn
        """
        if self.game_over:
            raise ValueError("Game is already over")
            
        if self.current_player != self.ai_symbol:
            raise ValueError("It's not AI's turn")
            
        # Get and make the best move
        move = self.get_best_move(self.board)
        self.board[move] = self.ai_symbol
        
        # Update game state
        game_state = self.get_game_state()
        if game_state['status'] != 'ongoing':
            self.game_over = True
            if game_state['status'] == 'win':
                self.winner = game_state['winner']
        else:
            self.current_player = self.human_symbol
            
        return move, self.serialize_game_state()
    
    def serialize_game_state(self):
        """
        Convert the game state to a JSON-serializable format for frontend communication.
        
        Returns:
            dict: Serialized game state
        """
        game_state = self.get_game_state()
        return {
            'board': self.board,
            'currentPlayer': self.current_player,
            'gameOver': self.game_over,
            'status': game_state['status'],
            'winner': game_state.get('winner'),
            'humanSymbol': self.human_symbol,
            'aiSymbol': self.ai_symbol
        }

# Standalone test for the Tic-Tac-Toe service
if __name__ == "__main__":
    def print_board(board):
        """Helper function to print the board in a readable format"""
        for i in range(0, 9, 3):
            print(f"{board[i] or ' '} | {board[i+1] or ' '} | {board[i+2] or ' '}")
            if i < 6:
                print("---------")
    
    # Test scenarios
    print("=== Testing TicTacToe Game Service ===")
    
    # Initialize game
    game = TicTacToeGame()
    print("Initial board:")
    print_board(game.board)
    
    # Test scenario: AI blocking human win
    print("\nScenario 1: AI blocking human win")
    game.reset_game()
    game.make_move(0)  # Human: top-left
    game.ai_move()     # AI's move
    game.make_move(1)  # Human: top-middle
    move, state = game.ai_move()  # AI should block by playing top-right
    print(f"AI moved at position {move}")
    print_board(game.board)
    
    # Test scenario: AI winning when possible
    print("\nScenario 2: AI winning when possible")
    game.reset_game()
    
    # Setup a board where AI can win in one move
    game.board = [
        'X', None, None,
        'O', 'O', None,
        'X', None, None
    ]
    game.current_player = game.ai_symbol
    
    move, state = game.ai_move()  # AI should win by playing middle-right
    print(f"AI moved at position {move}")
    print_board(game.board)
    print(f"Game state: {state['status']}, Winner: {state.get('winner')}")
    
    # Test scenario: Force a draw with optimal play
    print("\nScenario 3: Optimal play leading to draw")
    game.reset_game()
    
    # Human plays center
    game.make_move(4)
    print("Human plays center:")
    print_board(game.board)
    
    # AI responds (should be a corner)
    move, _ = game.ai_move()
    print(f"AI responds at position {move}:")
    print_board(game.board)
    
    print("\nContinuing with optimal play from both sides...")
    
    # Continue with a sequence that should lead to a draw with optimal play
    try:
        # Optimal sequence that should lead to draw
        moves = [0, 2, 6]
        for move in moves:
            if game.board[move] is None and not game.game_over:
                print(f"Human plays position {move}")
                game.make_move(move)
                print_board(game.board)
                
                if not game.game_over:
                    ai_move, state = game.ai_move()
                    print(f"AI responds at position {ai_move}")
                    print_board(game.board)
                    
                    if game.game_over:
                        print(f"Game over: {state['status']}, Winner: {state.get('winner')}")
                        break
            else:
                # Skip occupied positions
                continue
                
        # Check final state
        final_state = game.get_game_state()
        print(f"Final game state: {final_state['status']}")
        if 'winner' in final_state:
            print(f"Winner: {final_state['winner']}")
            
    except ValueError as e:
        print(f"Error: {e}")
