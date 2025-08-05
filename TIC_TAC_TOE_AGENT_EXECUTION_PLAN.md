# Detailed Execution Plan for Tic-Tac-Toe Agent Service

## Overview
This execution plan outlines the structure and implementation strategy for a self-contained Tic-Tac-Toe agent service designed to integrate seamlessly with the existing chatbot environment. The service will enable human vs. AI gameplay through a responsive UI while maintaining the current system's functionality and user experience.

## Phase 1: Backend Core Logic (tic_tac_toe_service.py)

### 1.1 File Creation
- Create a new Python file `tic_tac_toe_service.py` in the `backend/services` directory
- Import necessary modules (`random`, `copy`, `json`)
- Set up proper docstrings and module-level documentation

### 1.2 Game State Representation
- Define the `TicTacToeGame` class with the following attributes:
  - `board`: 1D list of 9 elements (initially filled with `None` values)
  - `human_symbol`: String (default: 'X')
  - `ai_symbol`: String (default: 'O')
  - `current_player`: String to track whose turn it is
  - `game_over`: Boolean to track if the game has ended
  - `winner`: String to store the winner (None if draw or ongoing)

### 1.3 Player Symbols
- Implement constructor to accept custom symbols:
  ```python
  def __init__(self, human_symbol='X', ai_symbol='O'):
      self.human_symbol = human_symbol
      self.ai_symbol = ai_symbol
      # Initialize other attributes
  ```
- Add validation to ensure symbols are different and single characters

### 1.4 Win/Draw Detection
- Implement the following methods:
  - `check_win(self, board, player)`: Check if the given player has won
    - Check all rows, columns, and diagonals
    - Return `True` if the player has won, `False` otherwise
  - `check_draw(self, board)`: Check if the game is a draw
    - Return `True` if the board is full with no winner
  - `get_game_state(self)`: Return the current game state
    - Possible returns: `{'status': 'ongoing'}`, `{'status': 'win', 'winner': 'X'}`, `{'status': 'draw'}`

### 1.5 Minimax Algorithm Implementation
- Implement the core minimax algorithm:
  ```python
  def minimax(self, board, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
      # Base cases: check for terminal states
      # Recursive case: evaluate all possible moves
      # Return optimal score and move
  ```
- Add alpha-beta pruning for improved performance
- Implement helper function `get_best_move(self, board)` that calls minimax to find the optimal move

### 1.6 Game Control Functions
- `reset_game(self)`: Reset the board and game state
- `make_move(self, index)`: Process a human move at the given index
  - Validate the move is legal
  - Update the board
  - Check for game end conditions
  - Return updated game state
- `ai_move(self)`: Calculate and execute the AI's move
  - Use the minimax algorithm to find the best move
  - Update the board
  - Check for game end conditions
  - Return the move made and updated game state
- `get_available_moves(self, board)`: Return a list of empty cell indices
- `serialize_game_state(self)`: Convert game state to a JSON-serializable format for frontend communication

### 1.7 Standalone Test
- Include a `if __name__ == "__main__":` block to test the service
- Implement simple test scenarios:
  - AI playing optimally to win when possible
  - AI blocking human win attempts
  - AI forcing a draw when optimal play is maintained

## Phase 2: Frontend UI Development (React)

### 2.1 React App Structure
- Review existing React structure in `chatbot-ui`
- Identify integration points for the new game component
- Plan component hierarchy:
  - App (main container)
  - Sidebar (navigation)
  - TicTacToeGame (game board and controls)
  - GameModal (status and result display)

### 2.2 Theme Management
- Ensure theme management is consistent with the existing system
- Use React context or state management to maintain theme consistency
- Apply appropriate CSS classes based on theme
- Update localStorage logic to persist theme preferences

### 2.3 Sidebar Component Enhancement
- Create/update `Sidebar.jsx` to include:
  - Existing profile icon
  - Existing chat icon
  - New TicTacToe icon (positioned left of the theme toggle)
  - Theme toggle icon (preserve existing functionality)
- Implement onClick handlers for view switching
- Apply consistent styling for all icons
- Ensure sidebar is visible across all views

### 2.4 Game Component
- Create `TicTacToeGame.jsx` with the following structure:
  ```jsx
  const TicTacToeGame = () => {
    const [board, setBoard] = useState(Array(9).fill(null));
    const [gameStatus, setGameStatus] = useState({status: 'ongoing'});
    const [isHumanTurn, setIsHumanTurn] = useState(true);
    // Game state logic
    // Event handlers
    // Rendering logic
  }
  ```
- Implement UI elements:
  - 3x3 grid with clickable cells
  - Status display showing current player and game state
  - "Restart Game" button
  - Game result modal

### 2.5 Placeholder Components
- Create minimal placeholders that represent existing functionality:
  - `ChatPlaceholder.jsx`: Simple representation of the chat interface
  - `ProfilePlaceholder.jsx`: Simple representation of the user profile

### 2.6 Main Content Rendering
- Update `App.jsx` or create a `MainContent.jsx` component
- Implement view switching logic:
  ```jsx
  const [activeView, setActiveView] = useState('chat');
  // Rendering logic based on activeView
  ```
- Ensure smooth transitions between views
- Preserve state when switching views

## Phase 3: Frontend-Backend Integration & Communication

### 3.1 Backend Integration
- For development and demonstration:
  - Create `ticTacToeService.js` to interface with the Python backend
  - Implement functions for game initialization, move processing, and state retrieval
- Document how this would transition to actual API calls in production

### 3.2 Game Flow Logic
- In `TicTacToeGame.jsx`, implement the game flow:
  ```jsx
  const handleCellClick = async (index) => {
    // Validate move
    // Update UI
    // Make human move
    // Check game status
    // If game continues, trigger AI move
    // Update UI again
  }
  ```
- Manage turn-based gameplay
- Handle exceptions and edge cases

### 3.3 Status Display (Modals/Alerts)
- Create `GameModal.jsx` component for game status display
- Implement different modal content based on game state:
  - Win: "Congratulations! You won!"
  - Loss: "AI wins this round!"
  - Draw: "It's a draw!"
- Add animations for improved user experience

### 3.4 Styling & Responsiveness
- Apply Tailwind CSS for all components
- Implement responsive design principles:
  - Mobile-first approach
  - Flexible grid layouts
  - Appropriate spacing and typography
- Ensure consistent styling with the existing UI
- Test on multiple screen sizes

## Phase 4: Testing and Refinement

### 4.1 Unit Testing (Backend)
- Write unit tests for all backend functions:
  - Board state validation
  - Win/draw detection
  - Minimax algorithm correctness
  - Game state serialization

### 4.2 Functional Testing (Frontend)
- Test all UI elements:
  - Cell click handlers
  - Game restart functionality
  - View switching
  - Theme toggle
- Verify game logic:
  - Human moves properly registered
  - AI responds with valid moves
  - Win/draw conditions correctly identified

### 4.3 Integration Testing
- Test the complete game flow:
  - Starting a new game
  - Making moves
  - Ending a game
  - Restarting
- Verify seamless view switching without state loss
- Test theme persistence across views

### 4.4 Code Review & Documentation
- Thoroughly comment all code
- Create detailed API documentation for the `tic_tac_toe_service.py` module
- Document integration points for frontend and backend
- Add setup instructions for future developers

## Implementation Timeline

### Week 1: Backend Development
- Day 1-2: Create `tic_tac_toe_service.py` with core game logic
- Day 3-4: Implement minimax algorithm and game control functions
- Day 5: Write unit tests and standalone testing

### Week 2: Frontend Development
- Day 1-2: Create basic UI components and integrate with sidebar
- Day 3-4: Implement game board and user interactions
- Day 5: Style components and ensure responsiveness

### Week 3: Integration & Testing
- Day 1-2: Connect frontend and backend
- Day 3: Implement status displays and modals
- Day 4-5: Comprehensive testing and bug fixing

## Conclusion
This execution plan provides a comprehensive roadmap for developing and integrating a Tic-Tac-Toe game module into the existing chatbot environment. By following this structured approach, the implementation will maintain the integrity of the current system while adding new functionality in a modular way.
