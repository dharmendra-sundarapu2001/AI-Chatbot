import os
import time
from typing import Optional
from fastapi import FastAPI, Depends, Request, HTTPException, status, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import logging
from datetime import datetime

# Database and models
from database import get_db
from models import User, Thread, Chat

# Services
from services.userService import user_service
from services.threadService import thread_service
from services.chatService import chat_service
from services.sqlService import SQLService

# Import Tic-Tac-Toe router
from tic_tac_toe_routes import tic_tac_toe_router

# Set up logging - disable uvicorn access logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Disable uvicorn access logs
logging.getLogger("uvicorn.access").disabled = True

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str

class ThreadUpdateRequest(BaseModel):
    title: str


# --- Authentication Endpoints ---
@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    logger.info(f"🔐 SIGNUP - {user.email}")
    
    result = user_service.create_user(user.email, user.password, db)
    if "error" in result:
        logger.info(f"❌ SIGNUP FAILED - {user.email} | {result['error']}")
        return JSONResponse(status_code=result["status_code"], content={"detail": result["error"]})
    
    logger.info(f"✅ SIGNUP SUCCESS - {user.email}")
    return result

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user login"""
    logger.info(f"🔐 LOGIN - {user.email}")
    
    result = user_service.authenticate_user(user.email, user.password, db)
    if "error" in result:
        logger.info(f"❌ LOGIN FAILED - {user.email} | {result['error']}")
        return JSONResponse(status_code=result["status_code"], content={"detail": result["error"]})
    
    logger.info(f"✅ LOGIN SUCCESS - {user.email}")
    return result

@app.post("/auth/google")
async def auth_google(auth_request: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Authenticate with Google OAuth"""
    result = await user_service.authenticate_google_user(auth_request.id_token, db)
    if "error" in result:
        logger.info(f"❌ GOOGLE AUTH FAILED | {result['error']}")
        raise HTTPException(status_code=result["status_code"], detail=result["error"])
    
    logger.info(f"✅ GOOGLE AUTH SUCCESS - {result.get('email', 'Unknown')}")
    return result


# --- Chat Endpoints ---
@app.post("/chat")
async def chat(
    question: str = Form(""),
    file: Optional[UploadFile] = File(None),
    thread_id: Optional[int] = Form(None),
    model: str = Form("chatgpt"),
    x_user_email: str = Header(None),
    db: Session = Depends(get_db)
):
    """Process chat message with optional file upload"""
    start_time = time.time()
    
    # Log user input
    file_info = f" | File: {file.filename}" if file else ""
    logger.info(f"💬 USER INPUT - {x_user_email} | Model: {model}{file_info}")
    logger.info(f"   Question: '{question}'")

    # Verify user authorization
    user = user_service.verify_user_authorization(x_user_email, db)
    if not user:
        logger.info(f"❌ UNAUTHORIZED - {x_user_email}")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: User not found"})

    # Validate input
    if not question and not file:
        logger.info(f"❌ NO INPUT - {x_user_email}")
        raise HTTPException(status_code=400, detail="Either a text message or a file must be provided.")

    # Handle thread creation/retrieval
    thread_id_val = thread_id
    if thread_id_val is None:
        initial_title = question[:50] if question else (file.filename[:50] if file else "New Chat")
        new_thread = thread_service.create_thread(user.id, initial_title, db)
        thread_id_val = new_thread.id
        logger.info(f"📝 NEW THREAD - {x_user_email} | ID: {thread_id_val} '")
    else:
        logger.info(f"📖 THREAD {thread_id_val} - {x_user_email}")

    # Process file upload
    file_data = None
    if file:
        file_data = await chat_service.process_file_upload(file, x_user_email, thread_id_val, user.id, db)

    # Create appropriate question for LLM
    question_to_llm = question
    if file_data and not question:
        # For invoices, let the chat service handle the question generation
        if not (file_data.get("is_invoice")):
            if file_data["file_mimetype"] == "application/pdf":
                question_to_llm = f"User provided a PDF file: {file_data['filename']}."
            else:
                question_to_llm = f"User provided a file: {file_data['filename']}."

    # Save user message
    user_chat = chat_service.save_user_message(question, thread_id_val, file_data, db, x_user_email)

    # Get recent chat history for context
    recent_chats = chat_service.get_recent_chat_history(thread_id_val, 5, db, x_user_email)

    # Build messages for LLM
    langchain_messages = chat_service.build_langchain_messages(recent_chats, question_to_llm, file_data, x_user_email, thread_id_val)

    # Process LLM response
    llm_result = await chat_service.process_llm_response(langchain_messages, model, x_user_email, thread_id_val, question)

    # Save bot response
    bot_chat = chat_service.save_bot_message(
        llm_result["bot_message_content"],
        thread_id_val,
        llm_result["image_data_base64"],
        llm_result["image_mime_type"],
        llm_result["video_data_base64"],
        llm_result["video_mime_type"],
        db,
        x_user_email
    )

    end_time = time.time()
    
    # Log bot response
    response_summary = f"Text: {len(llm_result['bot_message_content'])} chars"
    if llm_result['image_data_base64']:
        response_summary += " | Image: ✅"
    if llm_result['video_data_base64']:
        response_summary += " | Video: ✅"
    
    logger.info(f"🤖 BOT RESPONSE - {x_user_email}")
    logger.info(f"   Message: '{llm_result['bot_message_content']}'") # This line logs the actual content


    # Return response
    response_data = {
        "answer": llm_result["bot_message_content"],
        "thread_id": thread_id_val,
        "image_data_base64": llm_result["image_data_base64"],
        "image_mime_type": llm_result["image_mime_type"],
        "video_data_base64": llm_result["video_data_base64"],
        "video_mime_type": llm_result["video_mime_type"],
        "websearch_info": llm_result.get("websearch_info")
    }
    
    # Include file information if a file was uploaded
    if file_data:
        response_data["filename"] = file_data.get("filename")
        response_data["file_type"] = file_data.get("file_type")
        response_data["mime_type"] = file_data.get("mime_type")
    
    return JSONResponse(content=response_data)

@app.get("/chats")
def get_chats(request: Request, db: Session = Depends(get_db)):
    """Get all chats for the authenticated user"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    
    # Get user's thread IDs
    user_thread_ids = thread_service.get_thread_ids_for_user(user.id, db)
    
    # Get all chats for these threads
    chats_data = chat_service.get_all_user_chats(user_thread_ids, db, user_email)
    
    return jsonable_encoder(chats_data)


# --- Thread Endpoints ---
@app.get("/threads")
def get_threads(request: Request, db: Session = Depends(get_db)):
    """Get all threads for the authenticated user"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    
    threads = thread_service.get_user_threads(user.id, db)
    
    return jsonable_encoder(threads)

@app.get("/threads/{thread_id}/chats")
def get_thread_chats(thread_id: int, request: Request, db: Session = Depends(get_db)):
    """Get all chats for a specific thread"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    # Verify thread belongs to user
    thread = thread_service.get_thread_by_id(thread_id, user.id, db)
    if not thread:
        return JSONResponse(status_code=404, content={"detail": "Thread not found or access denied"})

    # Get chats for this thread
    chat_data = chat_service.get_thread_chats(thread_id, db, user_email)
    
    return jsonable_encoder(chat_data)

@app.put("/threads/{thread_id}")
def update_thread(thread_id: int, thread_update: ThreadUpdateRequest, request: Request, db: Session = Depends(get_db)):
    """Update thread title"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    # Get old title for logging
    old_thread = thread_service.get_thread_by_id(thread_id, user.id, db)
    old_title = old_thread.title if old_thread else "Unknown"

    updated_thread = thread_service.update_thread_title(thread_id, thread_update.title, user.id, db)
    
    if updated_thread:
        logger.info(f"✏️ THREAD RENAMED - {user_email} | ID: {thread_id} | '{old_title}' → '{thread_update.title}'")
    
    return jsonable_encoder(updated_thread)

@app.put("/threads/{thread_id}/pin")
def pin_thread(thread_id: int, request: Request, db: Session = Depends(get_db)):
    """Toggle pin status of a thread"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    updated_thread = thread_service.toggle_pin_thread(thread_id, user.id, db)
    
    if updated_thread:
        action = "PINNED" if updated_thread.pinned else "UNPINNED"
        logger.info(f"📌 THREAD {action} - {user_email} | ID: {thread_id} | '{updated_thread.title}'")
    
    return jsonable_encoder(updated_thread)

@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: int, request: Request, db: Session = Depends(get_db)):
    """Delete a thread and all its chats"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    thread = thread_service.get_thread_by_id(thread_id, user.id, db)
    if thread:
        thread_title = thread.title
        success = thread_service.delete_thread(thread_id, user.id, db)
        
        if success:
            logger.info(f"🗑️ THREAD DELETED - {user_email} | ID: {thread_id} | Title: '{thread_title}'")
            return JSONResponse(content={"success": True, "message": "Thread deleted successfully"})
        else:
            logger.error(f"❌ THREAD DELETE FAILED - {user_email} | ID: {thread_id}")
            raise HTTPException(status_code=500, detail="Failed to delete thread")
    else:
        raise HTTPException(status_code=404, detail="Thread not found")

# RAG Management Endpoints
@app.get("/rag/stats")
def get_rag_stats(request: Request, db: Session = Depends(get_db)):
    """Get statistics about the RAG vector store"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    try:
        stats = chat_service.rag_service.get_collection_stats()
        logger.info(f"📊 RAG STATS REQUESTED - {user_email}")
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"❌ RAG STATS ERROR - {user_email} | Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG stats: {str(e)}")

@app.delete("/rag/files/{filename}")
def delete_rag_file(filename: str, request: Request, db: Session = Depends(get_db)):
    """Delete a specific file from the RAG vector store"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    try:
        result = chat_service.rag_service.delete_file_from_vectorstore(filename, user_email)
        
        if result["status"] == "success":
            logger.info(f"🗑️ RAG FILE DELETED - {user_email} | Filename: {filename}")
            return JSONResponse(content=result)
        elif result["status"] == "not_found":
            logger.warning(f"🔍 RAG FILE NOT FOUND - {user_email} | Filename: {filename}")
            raise HTTPException(status_code=404, detail=result["message"])
        else:
            logger.error(f"❌ RAG FILE DELETE ERROR - {user_email} | Filename: {filename} | Error: {result['message']}")
            raise HTTPException(status_code=500, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RAG FILE DELETE ERROR - {user_email} | Filename: {filename} | Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file from RAG: {str(e)}")

@app.post("/rag/search")
def search_rag_content(query: str = Form(...), request: Request = None, db: Session = Depends(get_db)):
    """Search for relevant content in the RAG vector store"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    try:
        relevant_chunks = chat_service.rag_service.search_relevant_content(
            query=query, 
            user_email=user_email, 
            top_k=5
        )
        
        logger.info(f"🔍 RAG SEARCH - {user_email} | Query: '{query[:50]}...' | Results: {len(relevant_chunks)}")
        
        return JSONResponse(content={
            "query": query,
            "results": relevant_chunks,
            "total_results": len(relevant_chunks)
        })
        
    except Exception as e:
        logger.error(f"❌ RAG SEARCH ERROR - {user_email} | Query: '{query[:50]}...' | Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search RAG content: {str(e)}")

@app.get("/rag/documents")
def get_rag_documents(request: Request, db: Session = Depends(get_db)):
    """Get list of user's RAG documents from database"""
    user_email = request.headers.get("X-User-Email")
    
    user = user_service.verify_user_authorization(user_email, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    try:
        from models import RAGDocument
        
        rag_docs = db.query(RAGDocument).filter(RAGDocument.user_id == user.id).order_by(RAGDocument.timestamp.desc()).all()
        
        documents = []
        for doc in rag_docs:
            documents.append({
                "id": doc.id,
                "filename": doc.filename,
                "file_hash": doc.file_hash,
                "file_size": doc.file_size,
                "chunks_count": doc.chunks_count,
                "total_characters": doc.total_characters,
                "status": doc.status,
                "timestamp": doc.timestamp.isoformat()
            })
        
        logger.info(f"📋 RAG DOCUMENTS LIST - {user_email} | Count: {len(documents)}")
        
        return JSONResponse(content={
            "documents": documents,
            "total_count": len(documents)
        })
        
    except Exception as e:
        logger.error(f"❌ RAG DOCUMENTS LIST ERROR - {user_email} | Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG documents: {str(e)}")


# --- SQL Natural Language Query Endpoints ---

# Initialize SQL service
sql_service = SQLService()

@app.post("/sql/query")
async def natural_language_sql_query(
    question: str = Form(...),
    x_user_email: str = Header(None),
    db: Session = Depends(get_db)
):
    """Convert natural language question to SQL query and execute it"""
    start_time = time.time()
    
    logger.info(f"🔍 SQL QUERY REQUEST - {x_user_email}")
    logger.info(f"   Question: '{question}'")
    
    # Verify user authorization
    user = user_service.verify_user_authorization(x_user_email, db)
    if not user:
        logger.info(f"❌ UNAUTHORIZED - {x_user_email}")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: User not found"})
    
    try:
        # Process the natural language question
        result = sql_service.answer_natural_language_question(question)
        
        processing_time = round(time.time() - start_time, 2)
        
        if result["status"] == "success":
            logger.info(f"✅ SQL QUERY SUCCESS - {x_user_email} | Time: {processing_time}s")
            logger.info(f"   Query: {result['sql_query']}")
            logger.info(f"   Results: {result['row_count']} rows")
            
            return JSONResponse(content={
                "status": "success",
                "question": result["question"],
                "sql_query": result["sql_query"],
                "data": result["data"],
                "row_count": result["row_count"],
                "natural_language_answer": result["natural_language_answer"],
                "processing_time": f"{processing_time}s"
            })
        else:
            logger.error(f"❌ SQL QUERY ERROR - {x_user_email} | Error: {result.get('message', 'Unknown error')}")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result.get("message", "Failed to process SQL query")
                }
            )
            
    except Exception as e:
        logger.error(f"❌ SQL QUERY EXCEPTION - {x_user_email} | Error: {e}")
        raise HTTPException(status_code=500, detail=f"SQL query processing failed: {str(e)}")

@app.get("/sql/schema")
async def get_database_schema(
    x_user_email: str = Header(None),
    db: Session = Depends(get_db)
):
    """Get the dvdrental database schema"""
    logger.info(f"📊 SCHEMA REQUEST - {x_user_email}")
    
    # Verify user authorization
    user = user_service.verify_user_authorization(x_user_email, db)
    if not user:
        logger.info(f"❌ UNAUTHORIZED - {x_user_email}")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: User not found"})
    
    try:
        schema = sql_service.get_database_schema()
        logger.info(f"✅ SCHEMA SUCCESS - {x_user_email}")
        
        return JSONResponse(content={
            "status": "success",
            "schema": schema
        })
        
    except Exception as e:
        logger.error(f"❌ SCHEMA ERROR - {x_user_email} | Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database schema: {str(e)}")

@app.get("/sql/samples")
async def get_sample_questions(
    x_user_email: str = Header(None),
    db: Session = Depends(get_db)
):
    """Get sample questions users can ask"""
    logger.info(f"💡 SAMPLE QUESTIONS REQUEST - {x_user_email}")
    
    # Verify user authorization
    user = user_service.verify_user_authorization(x_user_email, db)
    if not user:
        logger.info(f"❌ UNAUTHORIZED - {x_user_email}")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: User not found"})
    
    try:
        samples = sql_service.get_sample_questions()
        logger.info(f"✅ SAMPLE QUESTIONS SUCCESS - {x_user_email}")
        
        return JSONResponse(content={
            "status": "success",
            "sample_questions": samples
        })
        
    except Exception as e:
        logger.error(f"❌ SAMPLE QUESTIONS ERROR - {x_user_email} | Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sample questions: {str(e)}")

@app.get("/sql/test")
async def test_sql_connection(
    x_user_email: str = Header(None),
    db: Session = Depends(get_db)
):
    """Test connection to dvdrental database"""
    logger.info(f"🔧 SQL CONNECTION TEST - {x_user_email}")
    
    # Verify user authorization
    user = user_service.verify_user_authorization(x_user_email, db)
    if not user:
        logger.info(f"❌ UNAUTHORIZED - {x_user_email}")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: User not found"})
    
    try:
        result = sql_service.test_connection()
        
        if result["status"] == "success":
            logger.info(f"✅ SQL CONNECTION SUCCESS - {x_user_email}")
        else:
            logger.error(f"❌ SQL CONNECTION FAILED - {x_user_email}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"❌ SQL CONNECTION EXCEPTION - {x_user_email} | Error: {e}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")

# Register Tic-Tac-Toe router with FastAPI
app.include_router(tic_tac_toe_router)

logger.info("✅ Tic-Tac-Toe game API endpoints registered")
   