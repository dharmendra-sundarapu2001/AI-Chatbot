from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from services.tic_tac_toe_service import TicTacToeGame
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Create a router for Tic-Tac-Toe game endpoints
tic_tac_toe_router = APIRouter(prefix="/tic-tac-toe", tags=["tic-tac-toe"])

# Store game instances by user
game_instances = {}

def get_user_game(user_email):
    """
    Get or create a game instance for a user
    
    Args:
        user_email: The user's email as identifier
        
    Returns:
        TicTacToeGame: The user's game instance
    """
    if user_email not in game_instances:
        game_instances[user_email] = TicTacToeGame()
    return game_instances[user_email]

class MoveRequest(BaseModel):
    index: int

@tic_tac_toe_router.post("/reset")
async def reset_game(x_user_email: str = Header(None)):
    """Reset the game and return the initial state"""
    if not x_user_email or x_user_email == "guest@example.com":
        logger.info(f"❌ TIC-TAC-TOE RESET UNAUTHORIZED - Using default email")
        raise HTTPException(status_code=401, detail="Please log in to play. User not properly authenticated.")
    
    game = get_user_game(x_user_email)
    game_state = game.reset_game()
    
    logger.info(f"🎮 TIC-TAC-TOE GAME RESET - {x_user_email}")
    return JSONResponse(content=game_state)

@tic_tac_toe_router.post("/move")
async def make_move(move_data: MoveRequest, x_user_email: str = Header(None)):
    """Process a player's move"""
    if not x_user_email or x_user_email == "guest@example.com":
        logger.info(f"❌ TIC-TAC-TOE MOVE UNAUTHORIZED - Using default email")
        raise HTTPException(status_code=401, detail="Please log in to play. User not properly authenticated.")
    
    try:
        game = get_user_game(x_user_email)
        game_state = game.make_move(move_data.index)
        logger.info(f"🎮 TIC-TAC-TOE PLAYER MOVE - {x_user_email} | Position: {move_data.index}")
        return JSONResponse(content=game_state)
    except ValueError as e:
        logger.warning(f"⚠️ TIC-TAC-TOE INVALID MOVE - {x_user_email} | Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@tic_tac_toe_router.post("/ai-move")
async def ai_move(x_user_email: str = Header(None)):
    """Make AI's move and return the updated game state"""
    if not x_user_email or x_user_email == "guest@example.com":
        logger.info(f"❌ TIC-TAC-TOE AI MOVE UNAUTHORIZED - Using default email")
        raise HTTPException(status_code=401, detail="Please log in to play. User not properly authenticated.")
    
    try:
        game = get_user_game(x_user_email)
        move_position, game_state = game.ai_move()
        logger.info(f"🤖 TIC-TAC-TOE AI MOVE - {x_user_email} | Position: {move_position}")
        return JSONResponse(content=game_state)
    except ValueError as e:
        logger.warning(f"⚠️ TIC-TAC-TOE INVALID AI MOVE - {x_user_email} | Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
