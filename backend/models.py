from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    threads = relationship("Thread", back_populates="user", cascade="all, delete-orphan")
    rag_documents = relationship("RAGDocument", back_populates="user", cascade="all, delete-orphan")

class Thread(Base):
    __tablename__ = "threads"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, default="New Chat")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pinned = Column(Boolean, default=False, nullable=False)
    user = relationship("User", back_populates="threads")
    chats = relationship("Chat", back_populates="thread", cascade="all, delete-orphan")
    rag_documents = relationship("RAGDocument", back_populates="thread", cascade="all, delete-orphan")

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String, nullable=False)  # "user" or "bot"
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    thread_id = Column(Integer, ForeignKey("threads.id"), nullable=False)
    image_data_base64 = Column(Text, nullable=True)  # For storing base64 image data
    video_data_base64 = Column(Text, nullable=True)  # For storing base64 video data
    image_mime_type = Column(String, nullable=True)  # For storing image MIME type
    video_mime_type = Column(String, nullable=True)  # For storing video MIME type
    filename = Column(String, nullable=True)  # For storing uploaded file name
    file_type = Column(String, nullable=True)  # For storing file type (pdf, docx, image, video, etc.)

    # Relationship
    thread = relationship("Thread", back_populates="chats")

class RAGDocument(Base):
    __tablename__ = "rag_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False, index=True)
    file_hash = Column(String, nullable=False, index=True)  # MD5 hash of file content (removed unique constraint)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    thread_id = Column(Integer, ForeignKey("threads.id"), nullable=False)  # Thread-specific files
    file_size = Column(Integer, nullable=False)  # File size in bytes
    file_type = Column(String, nullable=False, default="unknown")  # File type (pdf, docx, csv, etc.)
    chunks_count = Column(Integer, nullable=False)  # Number of chunks created
    total_characters = Column(Integer, nullable=False)  # Total characters in the document
    status = Column(String, nullable=False, default="processed")  # processed, error, deleted
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="rag_documents")
    thread = relationship("Thread")