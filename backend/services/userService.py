from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from passlib.hash import bcrypt
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os

from models import User


class UserService:
    def __init__(self):
        self.ALLOWED_EMAIL_DOMAIN = os.getenv("ALLOWED_EMAIL_DOMAIN", "@amzur.com")
        self.GOOGLE_CLIENT_ID = os.getenv("VITE_GOOGLE_CLIENT_ID")
        
    def create_user(self, email: str, password: str, db: Session) -> dict:
        """Create a new user with email and password"""
        if not email.endswith(self.ALLOWED_EMAIL_DOMAIN):
            return {"error": f"Only {self.ALLOWED_EMAIL_DOMAIN} emails allowed", "status_code": 400}
            
        hashed_password = bcrypt.hash(password)
        db_user = User(email=email, password=hashed_password)
        
        try:
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            return {"message": "Signup successful"}
        except IntegrityError:
            db.rollback()
            return {"error": "Email already registered", "status_code": 400}
    
    def authenticate_user(self, email: str, password: str, db: Session) -> dict:
        """Authenticate user with email and password"""
        db_user = db.query(User).filter(User.email == email).first()
        
        if not db_user or not bcrypt.verify(password, db_user.password):
            return {"error": "Invalid credentials", "status_code": 401}
            
        if not email.endswith(self.ALLOWED_EMAIL_DOMAIN):
            return {"error": f"Only {self.ALLOWED_EMAIL_DOMAIN} emails allowed", "status_code": 400}
            
        return {"message": "Login successful"}
    
    async def authenticate_google_user(self, id_token_str: str, db: Session) -> dict:
        """Authenticate user with Google OAuth token"""
        try:
            idinfo = id_token.verify_oauth2_token(
                id_token_str, 
                google_requests.Request(), 
                self.GOOGLE_CLIENT_ID
            )
            email = idinfo.get("email")

            if not email or not email.endswith(self.ALLOWED_EMAIL_DOMAIN):
                return {
                    "error": f"Access denied. Only users with {self.ALLOWED_EMAIL_DOMAIN} email addresses are allowed.",
                    "status_code": 403
                }

            user = db.query(User).filter(User.email == email).first()
            if not user:
                user = User(email=email, password="")  # Google auth users don't have a password
                db.add(user)
                db.commit()
                db.refresh(user)

            return {"email": email, "message": "Google authentication successful"}
            
        except ValueError as e:
            return {
                "error": f"Google authentication failed: Invalid token or client ID mismatch. ({str(e)})",
                "status_code": 401
            }
        except Exception as e:
            print(f"An unexpected error occurred during Google token verification: {e}")
            return {
                "error": "Internal server error during authentication.",
                "status_code": 500
            }
    
    def get_user_by_email(self, email: str, db: Session) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email).first()
    
    def verify_user_authorization(self, user_email: str, db: Session) -> Optional[User]:
        """Verify user authorization and return user object"""
        if not user_email:
            return None
        
        user = db.query(User).filter(User.email == user_email).first()
        return user


# Create a singleton instance
user_service = UserService()
