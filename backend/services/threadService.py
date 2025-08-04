"""
Thread Service Module
Handles thread management operations including creation, update, pin/unpin, and deletion
"""
from typing import List
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from fastapi.encoders import jsonable_encoder

from models import Thread, User


class ThreadService:
    def __init__(self):
        pass
    
    def create_thread(self, user_id: int, title: str, db: Session) -> Thread:
        """Create a new thread for a user"""
        new_thread = Thread(user_id=user_id, title=title)
        db.add(new_thread)
        db.commit()
        db.refresh(new_thread)
        return new_thread
    
    def get_user_threads(self, user_id: int, db: Session) -> List[Thread]:
        """Get all threads for a user, ordered by pinned status and timestamp"""
        threads = (
            db.query(Thread)
            .filter(Thread.user_id == user_id)
            .order_by(Thread.pinned.desc(), Thread.timestamp.desc())
            .all()
        )
        return threads
    
    def get_thread_by_id(self, thread_id: int, user_id: int, db: Session) -> Thread:
        """Get a specific thread by ID, ensuring it belongs to the user"""
        thread = (
            db.query(Thread)
            .filter(Thread.id == thread_id, Thread.user_id == user_id)
            .first()
        )
        return thread
    
    def update_thread_title(self, thread_id: int, new_title: str, user_id: int, db: Session) -> Thread:
        """Update thread title"""
        thread = self.get_thread_by_id(thread_id, user_id, db)
        if not thread:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        
        thread.title = new_title
        db.commit()
        db.refresh(thread)
        return thread
    
    def toggle_pin_thread(self, thread_id: int, user_id: int, db: Session) -> Thread:
        """Toggle pin status of a thread"""
        thread = self.get_thread_by_id(thread_id, user_id, db)
        if not thread:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        
        thread.pinned = not thread.pinned
        db.commit()
        db.refresh(thread)
        return thread
    
    def delete_thread(self, thread_id: int, user_id: int, db: Session) -> dict:
        """Delete a thread and all its associated chats"""
        thread = self.get_thread_by_id(thread_id, user_id, db)
        if not thread:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        
        db.delete(thread)
        db.commit()
        return {"message": "Thread deleted successfully"}
    
    def get_thread_ids_for_user(self, user_id: int, db: Session) -> List[int]:
        """Get all thread IDs for a user (useful for chat filtering)"""
        thread_ids = db.query(Thread.id).filter(Thread.user_id == user_id).all()
        return [thread_id[0] for thread_id in thread_ids]


# Create a singleton instance
thread_service = ThreadService()
