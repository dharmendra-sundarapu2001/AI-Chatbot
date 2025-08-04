from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import logging
from models import Base, User, Chat, Thread, RAGDocument  # Import all models

# Set up logging
logger = logging.getLogger(__name__)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

# Create engine - PostgreSQL for main application data
engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    """Dependency function to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize the database and create all tables"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ PostgreSQL database tables created successfully")
        logger.info(f"🔗 Database URL: {DATABASE_URL}")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise

# Initialize database on import
init_database()
