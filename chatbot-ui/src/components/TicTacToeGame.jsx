import React, { useState, useEffect } from 'react';
import { getAuthHeaders, getBaseApiUrl } from '../utils/authUtils';


// Define custom animations - these will be injected into the page head
const customStyles = `
@keyframes bounce-once {
  0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-20px); }
  60% { transform: translateY(-10px); }
}


@keyframes shake {
  0%, 100% { transform: translateX(0); }
  10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
  20%, 40%, 60%, 80% { transform: translateX(5px); }
}


@keyframes fade-in {
  from { opacity: 0; transform: scale(0.9); }
  to { opacity: 1; transform: scale(1); }
}


.animate-bounce-once {
  animation: bounce-once 1s ease;
}


.animate-shake {
  animation: shake 0.5s ease-in-out;
}


.animate-fade-in {
  animation: fade-in 0.3s ease-out;
}
`;


/**
 * TicTacToeGame Component - Renders a Tic-Tac-Toe game board and manages game state
 * @returns {JSX.Element} The game component
 */
const TicTacToeGame = () => {
  // Game state
  const [board, setBoard] = useState(Array(9).fill(null));
  const [gameStatus, setGameStatus] = useState({ status: 'idle' });
  const [isHumanTurn, setIsHumanTurn] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalMessage, setModalMessage] = useState('');
  const [gameStarted, setGameStarted] = useState(false);
  const [lastMove, setLastMove] = useState(null);
  
  // Scoreboard state
  const [scores, setScores] = useState({
    wins: 0,
    losses: 0,
    draws: 0
  });
  
  // Inject custom animations
  useEffect(() => {
    // Create style element
    const styleElement = document.createElement('style');
    styleElement.type = 'text/css';
    styleElement.appendChild(document.createTextNode(customStyles));
    
    // Inject into the head
    document.head.appendChild(styleElement);
    
    // Clean up on component unmount
    return () => {
      document.head.removeChild(styleElement);
    };
  }, []);
  
  // Player symbols
  const humanSymbol = 'X';
  const aiSymbol = 'O';


  // Start a new game
  const startGame = async () => {
    setIsLoading(true);
    try {
      // Check if we have a proper user email from App.jsx
      const appEmail = localStorage.getItem('loggedInEmail');
      if (!appEmail || appEmail === 'guest@example.com') {
        throw new Error('Please log in first to play the game.');
      }
      
      // Ensure we're using the same email in headers as the one used for login
      const headers = {
        'Content-Type': 'application/json',
        'X-User-Email': appEmail
      };

      // Call the backend API to reset the game
      const response = await fetch(`${getBaseApiUrl()}/tic-tac-toe/reset`, {
        method: 'POST',
        headers: headers
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Authentication error. Please log in to play the game.');
      }
      
      const gameState = await response.json();
      console.log("Game started with state:", gameState);
      
      // Reset all game state
      setBoard(Array(9).fill(null)); // Force clear the board first
      setGameStatus({ status: 'ongoing' });
      setIsHumanTurn(true);
      setShowModal(false);
      setGameStarted(true);
      setLastMove(null);
      
    } catch (error) {
      console.error('Failed to start game:', error);
      // Show error message to user
      setModalMessage(error.message || 'Authentication error. Please log in to play the game.');
      setShowModal(true);
      setGameStarted(false);
    }
    setIsLoading(false);
  };


  // Reset game state
  const resetGame = async () => {
    setIsLoading(true);
    try {
      // Call the backend API to reset the game
      const response = await fetch(`${getBaseApiUrl()}/tic-tac-toe/reset`, {
        method: 'POST',
        headers: getAuthHeaders()
      });
      
      const gameState = await response.json();
      console.log("Game reset with state:", gameState);
      
      setBoard(Array(9).fill(null));
      setGameStatus({ status: 'ongoing' });
      setIsHumanTurn(true);
      setShowModal(false);
      setLastMove(null);
      
    } catch (error) {
      console.error('Failed to reset game:', error);
      // Fallback to local reset if API call fails
      setBoard(Array(9).fill(null));
      setGameStatus({ status: 'ongoing' });
      setIsHumanTurn(true);
      setShowModal(false);
      setLastMove(null);
    }
    setIsLoading(false);
  };


  // Request AI move
  const requestAiMove = async () => {
    try {
      // Check if we have a proper user email from App.jsx
      const appEmail = localStorage.getItem('loggedInEmail');
      if (!appEmail || appEmail === 'guest@example.com') {
        throw new Error('Please log in first to play the game.');
      }
      
      // Ensure we're using the same email in headers as the one used for login
      const headers = {
        'Content-Type': 'application/json',
        'X-User-Email': appEmail
      };
      
      const aiResponse = await fetch(`${getBaseApiUrl()}/tic-tac-toe/ai-move`, {
        method: 'POST',
        headers: headers
      });
      
      if (!aiResponse.ok) {
        const errorData = await aiResponse.json();
        throw new Error(errorData.detail || 'Authentication error. Please log in to continue playing.');
      }
      
      const aiGameState = await aiResponse.json();
      console.log("AI move result:", aiGameState);
      
      // Find which cell was changed to determine AI's move
      const aiMoveIndex = aiGameState.board.findIndex((cell, index) => 
        cell === aiSymbol && board[index] !== aiSymbol
      );
      
      setBoard(aiGameState.board);
      setGameStatus({
        status: aiGameState.status,
        winner: aiGameState.winner
      });
      setLastMove(aiMoveIndex);
      
      // Check if game ended after AI move
      if (aiGameState.status !== 'ongoing') {
        showGameResult(aiGameState);
      } else {
        setIsHumanTurn(true);
      }
      
      console.log(`AI played at position ${aiMoveIndex}`);
      
    } catch (error) {
      console.error('Failed to get AI move:', error);
      
      // Show error message to user
      setModalMessage(error.message || 'Authentication error. Please log in to continue playing.');
      setShowModal(true);
      setGameStarted(false);
    }
    
    setIsLoading(false);
  };


  // Handle cell click
  const handleCellClick = async (index) => {
    // Ignore click if game hasn't started, cell is filled, game is over, or it's AI's turn
    if (!gameStarted || board[index] || gameStatus.status !== 'ongoing' || !isHumanTurn || isLoading) {
      return;
    }


    setIsLoading(true);
    setLastMove(index);


    try {
      // Check if we have a proper user email from App.jsx
      const appEmail = localStorage.getItem('loggedInEmail');
      if (!appEmail || appEmail === 'guest@example.com') {
        throw new Error('Please log in first to play the game.');
      }
      
      // Ensure we're using the same email in headers as the one used for login
      const headers = {
        'Content-Type': 'application/json',
        'X-User-Email': appEmail
      };
      
      // Make human move locally first for immediate feedback
      const newBoard = [...board];
      newBoard[index] = humanSymbol;
      setBoard(newBoard);
      
      console.log(`Human played at position ${index}`);
      
      // Call the backend API to register the human move
      const response = await fetch(`${getBaseApiUrl()}/tic-tac-toe/move`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ index })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Authentication error. Please log in to play the game.');
      }
      
      const gameState = await response.json();
      console.log("Human move result:", gameState);
      
      // Update board from server response
      setBoard(gameState.board);
      
      // Check if game ended after human move
      if (gameState.status !== 'ongoing') {
        setGameStatus({
          status: gameState.status,
          winner: gameState.winner
        });
        showGameResult(gameState);
        setIsLoading(false);
        return;
      }


      // AI's turn
      setIsHumanTurn(false);
      
      // Add a small delay before AI's move for better UX
      setTimeout(async () => {
        await requestAiMove();
      }, 500);
      
    } catch (error) {
      console.error('Failed to process move:', error);
      // Show error message to user
      setModalMessage(error.message || 'Authentication error. Please log in to continue playing.');
      setShowModal(true);
      setIsLoading(false);
      setIsHumanTurn(true);
      setGameStarted(false);
    }
  };


  // Display game result modal and update scores
  const showGameResult = (gameState) => {
    let message = '';
    
    if (gameState.status === 'win') {
      if (gameState.winner === humanSymbol) {
        message = 'Congratulations! You won! 🎉';
        setScores(prev => ({ ...prev, wins: prev.wins + 1 }));
      } else {
        message = 'AI wins this round! 🤖';
        setScores(prev => ({ ...prev, losses: prev.losses + 1 }));
      }
    } else if (gameState.status === 'draw') {
      message = "It's a draw! 🤝";
      setScores(prev => ({ ...prev, draws: prev.draws + 1 }));
    }
    
    setModalMessage(message);
    setShowModal(true);
  };


  // Effect for game result notification
  useEffect(() => {
    // When game is over, add confetti or shake effect
    if (gameStatus.status === 'win' || gameStatus.status === 'draw') {
      // Animation is handled via CSS in the modal
    }
  }, [gameStatus.status]);
  
  // Render the game board
  return (
    <div className="h-full w-full flex flex-col items-center justify-center overflow-hidden">
      <div className="relative flex flex-col items-center justify-center p-4 sm:p-5 md:p-6 w-full max-w-md mx-auto rounded-xl overflow-hidden" 
        style={{
          background: 'linear-gradient(135deg, #1a1c29 0%, #2a3045 100%)',
          boxShadow: '0 10px 20px rgba(0,0,0,0.3)',
          border: '1px solid #3a4055'
        }}>
        <div className="absolute inset-0 rounded-xl opacity-10" 
          style={{
            backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'60\' height=\'60\' viewBox=\'0 0 60 60\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cpath d=\'M54.627 0l.83.828-1.415 1.415L51.8 0h2.827zM5.373 0l-.83.828L5.96 2.243 8.2 0H5.374zM48.97 0l3.657 3.657-1.414 1.414L46.143 0h2.828zM11.03 0L7.372 3.657 8.787 5.07 13.857 0H11.03zm32.284 0L49.8 6.485 48.384 7.9l-7.9-7.9h2.83zM16.686 0L10.2 6.485 11.616 7.9l7.9-7.9h-2.83zm20.97 0l9.315 9.314-1.414 1.414L34.828 0h2.83zM22.344 0L13.03 9.314l1.414 1.414L25.172 0h-2.83zM32 0l12.142 12.142-1.414 1.414L30 .828 17.272 13.556l-1.414-1.414L28 0h4zM.284 0l28 28-1.414 1.414L0 2.544v2.83L25.456 30h-2.83L0 7.373v2.83L22.627 30h-2.83L0 12.202v2.83L19.8 30h-2.83L0 17.03v2.83L16.97 30h-2.83L0 21.86v2.83L14.142 30h-2.83L0 26.688v2.83L11.313 30h-2.83L0 30h60L48.687 18.687l2.83 2.83L60 30V27.17L42.142 9.313l-1.414 1.414L60 30V24.687L37.543 2.23 60 24.687V21.86L34.457 -3.172 60 21.86V17.03L31.37-8.04 60 17.03V12.203L28.284-13.313 60 12.203V7.373L25.2-18.656 60 7.373V4.544L22.113-23.87 60 4.544V0L0 0z\' fill=\'%239C92AC\' fill-opacity=\'0.1\' fill-rule=\'evenodd\'/%3E%3C/svg%3E")'
          }}>
        </div>
      
      <h1 className="relative text-2xl sm:text-3xl font-bold mb-4 sm:mb-6 text-white tracking-wide">
        Tic-Tac-Toe <span className="text-blue-400">Game</span>
      </h1>
      
      {/* Scoreboard */}
      <div className="relative flex justify-center mb-4 sm:mb-6 gap-3 sm:gap-6">
        <div className="flex flex-col items-center bg-green-900 bg-opacity-50 rounded-lg px-3 sm:px-5 py-2 sm:py-3 border border-green-500 transform transition-transform hover:scale-105">
          <span className="text-xs sm:text-sm text-green-300">Wins</span>
          <span className="text-xl sm:text-2xl font-bold text-green-300">{scores.wins}</span>
        </div>
        <div className="flex flex-col items-center bg-blue-900 bg-opacity-50 rounded-lg px-3 sm:px-5 py-2 sm:py-3 border border-blue-500 transform transition-transform hover:scale-105">
          <span className="text-xs sm:text-sm text-blue-300">Draws</span>
          <span className="text-xl sm:text-2xl font-bold text-blue-300">{scores.draws}</span>
        </div>
        <div className="flex flex-col items-center bg-red-900 bg-opacity-50 rounded-lg px-3 sm:px-5 py-2 sm:py-3 border border-red-500 transform transition-transform hover:scale-105">
          <span className="text-xs sm:text-sm text-red-300">Losses</span>
          <span className="text-xl sm:text-2xl font-bold text-red-300">{scores.losses}</span>
        </div>
      </div>
      
      {!gameStarted ? (
        // Show start game button when game hasn't started
        <div className="relative flex flex-col items-center z-10">
          <p className="mb-6 text-xl text-blue-200 font-medium">Ready to challenge the AI?</p>
          <button
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-xl font-bold rounded-lg
              hover:from-blue-700 hover:to-purple-700 transition-all duration-300 shadow-lg transform hover:scale-105
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
            onClick={startGame}
            disabled={isLoading}
          >
            {isLoading ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Starting...
              </span>
            ) : 'Start Game'}
          </button>
        </div>
      ) : (
        // Show game board when game has started
        <>
          {/* Game status */}
          <div className="relative mb-6 text-lg text-center z-10">
            {gameStatus.status === 'ongoing' ? (
              isHumanTurn ? (
                <div className="flex items-center justify-center bg-green-900 bg-opacity-50 py-2 px-4 rounded-lg border border-green-500">
                  <span className="h-4 w-4 bg-green-400 rounded-full animate-pulse mr-2"></span>
                  <span className="text-green-200 font-medium">Your turn (X)</span>
                </div>
              ) : (
                <div className="flex items-center justify-center bg-blue-900 bg-opacity-50 py-2 px-4 rounded-lg border border-blue-500">
                  <span className="h-4 w-4 bg-blue-400 rounded-full animate-pulse mr-2"></span>
                  <span className="text-blue-200 font-medium">AI thinking... (O)</span>
                </div>
              )
            ) : gameStatus.status === 'win' ? (
              <div className={`text-xl font-bold py-2 px-4 rounded-lg border 
                  ${gameStatus.winner === humanSymbol ? 'bg-green-900 bg-opacity-50 text-green-200 border-green-500' : 'bg-red-900 bg-opacity-50 text-red-200 border-red-500'}`}>
                Winner: {gameStatus.winner === humanSymbol ? 'You 🏆' : 'AI 🤖'}
              </div>
            ) : gameStatus.status === 'draw' ? (
              <div className="text-xl font-bold py-2 px-4 bg-blue-900 bg-opacity-50 text-blue-200 rounded-lg border border-blue-500">
                It's a draw! 🤝
              </div>
            ) : null}
          </div>
          
          {/* Game board */}
          <div className="relative grid grid-cols-3 gap-2 sm:gap-3 mb-6 sm:mb-8">
            {board.map((cell, index) => (
              <button
                key={index}
                className={`w-20 h-20 sm:w-24 sm:h-24 flex items-center justify-center text-4xl sm:text-5xl font-bold rounded-lg
                  shadow-lg transform transition-all duration-200
                  ${!isHumanTurn || cell !== null || gameStatus.status !== 'ongoing' || isLoading ? '' : 'hover:scale-105 hover:shadow-xl'}
                  ${lastMove === index ? 'ring-2 ring-yellow-400 animate-pulse' : ''}
                  ${cell === humanSymbol 
                    ? 'text-green-300 bg-green-900 bg-opacity-60 border border-green-500' 
                    : cell === aiSymbol 
                      ? 'text-blue-300 bg-blue-900 bg-opacity-60 border border-blue-500'
                      : 'bg-gray-800 bg-opacity-40 border border-gray-600 hover:border-gray-400'}
                  `}
                onClick={() => handleCellClick(index)}
                disabled={!isHumanTurn || cell !== null || gameStatus.status !== 'ongoing' || isLoading}
                aria-label={`Cell ${index}`}
                style={{
                  textShadow: cell ? '0 0 10px rgba(255,255,255,0.5)' : 'none'
                }}
              >
                {cell}
              </button>
            ))}
          </div>
          
          {/* Game control buttons */}
          <div className="relative flex space-x-4 z-10">
            <button
              className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold rounded-lg
                hover:from-blue-700 hover:to-indigo-700 transition-all duration-300 shadow-md transform hover:scale-105
                focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50"
              onClick={resetGame}
              disabled={isLoading}
            >
              {isLoading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Resetting...
                </span>
              ) : (
                <span className="flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Reset Game
                </span>
              )}
            </button>
          </div>
        </>
      )}
      
      {/* Result modal */}
      {showModal && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-70 z-50 backdrop-blur-sm">
          <div className={`bg-gray-900 border border-indigo-500 p-5 sm:p-8 rounded-xl shadow-2xl max-w-sm w-[90%] sm:w-full relative
            ${gameStatus.status === 'win' && gameStatus.winner === humanSymbol ? 'animate-bounce-once' : 
              gameStatus.status === 'win' ? 'animate-shake' : 'animate-fade-in'}`}
            style={{
              boxShadow: '0 0 30px rgba(79, 70, 229, 0.5)'
            }}>
            {/* Close button (X) in the top right */}
            <button 
              className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center text-gray-400 
                hover:text-white rounded-full hover:bg-gray-700 transition-colors"
              onClick={() => setShowModal(false)}
              aria-label="Close"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            
            <h2 className="text-2xl font-bold mb-4 text-white text-center">Game Over</h2>
            <p className="text-xl mb-6 text-center text-blue-200">{modalMessage}</p>
            
            {/* Game statistics */}
            <div className="flex justify-between mb-6 bg-gray-800 rounded-lg p-3">
              <div className="text-center px-2">
                <span className="text-sm text-green-400">Wins</span>
                <p className="text-xl font-bold text-green-300">{scores.wins}</p>
              </div>
              <div className="text-center px-2">
                <span className="text-sm text-blue-400">Draws</span>
                <p className="text-xl font-bold text-blue-300">{scores.draws}</p>
              </div>
              <div className="text-center px-2">
                <span className="text-sm text-red-400">Losses</span>
                <p className="text-xl font-bold text-red-300">{scores.losses}</p>
              </div>
            </div>
            
            <div className="flex justify-center">
              <button
                className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold rounded-lg
                  hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-lg
                  transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50"
                onClick={() => {
                  setShowModal(false);
                  resetGame();
                }}
              >
                Play Again
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
};


export default TicTacToeGame;
