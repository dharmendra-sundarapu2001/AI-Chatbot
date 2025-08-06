import os
import time
import requests
import base64
import logging
import re
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from models import Chat, Thread

# Import RAG service instead of PyPDF2
from services.rag.rag_service import RAGService
from services.sqlService import SQLService
from services.invoiceService import InvoiceService
from services.webSearchService import WebSearchService
from services.trip_planner import TripPlanningAgent
import io

# Set up logging configuration
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        # Load environment variables
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.CHATGPT_TOKEN = os.getenv("CHATGPT_TOKEN")
        self.HUGGING_FACE_API_TOKEN = os.getenv("HF_API_TOKEN")
        self.OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER")
        
        # Initialize RAG service
        self.rag_service = RAGService()
        
        # Initialize SQL service for natural language database queries
        self.sql_service = SQLService()
        
        # Initialize Invoice service for OCR and invoice processing
        self.invoice_service = InvoiceService()
        
        # Initialize Web Search service for real-time web data
        self.web_search_service = WebSearchService()
        
        # Initialize Trip Planning service for vacation planning
        self.trip_planning_agent = TripPlanningAgent()
        
        # Hugging Face endpoints
        self.HF_SD_MODEL_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        self.HF_VIDEO_MODEL_URL = "https://api-inference.huggingface.co/models/YOUR_TEXT_TO_VIDEO_MODEL_ENDPOINT"
        self.HF_HEADERS = {
            "Authorization": f"Bearer {self.HUGGING_FACE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # System prompt for AI behavior - MODIFIED FOR TOOL USAGE EMPHASIS
        self.SYSTEM_PROMPT = """You are a helpful AI assistant with the following capabilities:

1.  **Rich Text Formatting**: Use markdown formatting for better readability
2.  **Code Display**: Present code with proper syntax highlighting using code blocks
3.  **Structured Data**: Create tables for data presentation, comparisons, lists, etc.
4.  **Mathematical Content**: Render mathematical formulas and equations using LaTeX notation
5.  **Visual Content**: Generate images when requested using available tools
6.  **Document Analysis**: Analyze and provide insights from uploaded documents (PDFs, Excel, Word, etc.)
7.  **Organized Information**: Structure responses with headings, lists, and logical flow

CRITICAL: When asked to create images or videos, you MUST use the proper tool calls and NOT print function calls as text. 
DO NOT output text like "generate_image_tool(prompt='...')" or "default_api.generate_image_tool(...)". 
Instead, you MUST use the proper tool calling mechanism.

**Document Analysis Guidelines:**
- When provided with document content, analyze it comprehensively and provide detailed insights
- For Excel files, identify and explain data patterns, key metrics, and important information
- For PDFs and Word documents, summarize key points and extract relevant information
- Always reference the specific document when providing information from uploaded files
- Present data in well-formatted tables when appropriate
- Questions about file content (using phrases like "according to the file", "from the document", "based on the data", "which country/person/item", etc.) are ANALYSIS requests, NOT image generation requests

**Formatting Guidelines:**
- Use tables for any structured data (periodic tables, comparisons, schedules, etc.)
- Use code blocks with language specification for code
- Use LaTeX for mathematical expressions ($ for inline, $$ for block equations)
- Use appropriate headings to organize longer responses
- Use lists for step-by-step instructions or itemized information

**Image Generation:**
- When the user's request **explicitly and specifically** asks for a visual image, diagram, picture, drawing, or illustration to be created or generated, **YOU MUST, WITHOUT ANY TEXTUAL RESPONSE OR CLARIFYING QUESTIONS, IMMEDIATELY INVOKE THE `generate_image_tool`**.
- This includes ONLY direct image generation requests like "draw me a...", "create a picture of...", "generate an image of...", "make an illustration of...", "I want you to draw...", "please create a visual of...", or "design an image of...".
- **DO NOT** generate images for questions about document content, data analysis, file information, or general inquiries that happen to use words like "show me", "visualize", or "which". These are informational requests, not image creation requests.
- **ESPECIALLY DO NOT** generate images for questions that use phrases like "according to the file", "based on the document", "from the data", "which country", "what is", "who is", etc. These are data inquiry questions, NOT image creation requests.
- **CRITICAL INSTRUCTION FOR TOOL ARGUMENT (`prompt`):** When calling the `generate_image_tool`, the `prompt` argument **must be a highly detailed, descriptive, and creative textual representation of the desired image.**
    - **You MUST automatically infer and elaborate on any missing details from the user's request, creating a rich and imaginative scene. Focus heavily on positive descriptions for lighting and color.** For example, if the user says "beach view image", you should automatically generate a detailed prompt like: "An **ultra-photorealistic** image capturing a **breathtaking tropical beach at golden hour sunset**. The scene is bathed in **warm, glowing light**, casting **long, soft shadows**. The foreground features **crystal-clear, emerald-green ocean waves** gently lapping over **sparkling white sand** adorned with **iridescent seashells**. In the midground, a **richly textured, sun-kissed driftwood log** gleams beside the water's edge. The background showcases **lush, vibrant palm trees**, sharply silhouetted against a sky exploding with **fiery oranges, deep magentas, electric purples, and soft pastel pinks**. The vast ocean reflects the **brilliant, multi-hued sky**, creating a mesmerizing, **serene, and overwhelmingly colorful panorama**. Shot with a **wide-angle lens**, **high dynamic range (HDR)**."
    - **DO NOT provide any descriptive text *before* invoking the tool.** Your response should be solely the tool call, allowing the tool to execute.
    - **Always include specifics about:**
        - **Subject details:** (e.g., "a golden retriever puppy," "a majestic lion," "a vintage car")
        - **Actions/Activity:** (e.g., "drinking water from a clear glass bowl," "roaming the savanna," "parked on a cobblestone street")
        - **Setting/Background:** (e.g., "in a sunlit kitchen," "against a vibrant sunset," "in front of an old European building")
        - **Lighting:** (e.g., "**dynamic lighting**, **cinematic light**, **golden hour**, **soft natural light**, **dramatic backlighting**, **radiant sunbeams**, **glowing ambient light**, **sparkling highlights**")
        - **Style/Art Medium:** (e.g., "**ultra-photorealistic**, **high detail**, **8k photograph**, **film grain**, **award-winning photograph**, oil painting, digital art, dreamlike, anime style")
        - **Composition/Angle:** (e.g., "close-up shot," "wide angle," "from a low angle," "bird's-eye view")
        - **Colors/Mood:** (e.g., "**vibrant colors**, **rich hues**, **saturated tones**, peaceful mood, mysterious atmosphere, **multi-hued**")
- **After the `generate_image_tool` has successfully executed, then, and ONLY THEN, provide a concise, natural language textual description of the image that was generated. DO NOT mention that the image data is "provided separately" or explain the tool execution process. Simply describe the image content.**

**Video Generation:**
- When the user's request **clearly and unequivocally** asks for a video, animation, or anything that can be represented as moving visuals, **YOU MUST, WITHOUT ANY TEXTUAL RESPONSE OR CLARIFYING QUESTIONS, IMMEDIATELY INVOKE THE `generate_video_tool`**.
- This includes prompts like "generate a video", "create a video", "make a video animation", or "create a video of X".
- **CRITICAL INSTRUCTION FOR TOOL ARGUMENT (`prompt`):** Similar to image generation, when calling the `generate_video_tool`, the `prompt` argument **must be a highly detailed and descriptive textual representation of the desired video content.**
    - **You MUST automatically infer and elaborate on any missing details from the user's request, following the same detail guidelines as for image prompts.**
    - **DO NOT provide any descriptive text *before* invoking the tool.**
- **After the `generate_video_tool` has successfully executed, then, and ONLY THEN, provide a concise, natural language textual description of the video content and duration. DO NOT mention that the video data is "provided separately" or explain the tool execution process. Simply describe the video content and its approximate duration.**

Adapt your response format to best serve the user's specific request while maintaining clarity and proper formatting.
**Crucially, closely examine the provided "Recent conversation history" for context. If the user refers to a previously discussed person, place, or topic implicitly (e.g., "that person," "the place we just talked about," "him/her/it/they from before"), infer the specific entity from the history and continue the conversation about that entity. Always strive to maintain conversational flow and remember key details from prior turns.**
"""
        
        # Initialize LLMs with tools
        self._initialize_llms()
    
    def _map_file_type_for_ui(self, raw_file_type: str, filename: str, mime_type: str) -> str:
        """Map RAG service file types to UI-friendly file type identifiers"""
        # First check the raw file type from RAG
        if raw_file_type in ['xlsx', 'xls']:
            return "excel"
        elif raw_file_type == 'pdf':
            return "pdf"
        elif raw_file_type in ['docx', 'doc']:
            return "word"
        elif raw_file_type in ['pptx', 'ppt']:
            return "powerpoint"
        elif raw_file_type == 'csv':
            return "csv"
        elif raw_file_type in ['txt', 'md']:
            return "text"
        elif raw_file_type == 'json':
            return "json"
        elif raw_file_type in ['odt', 'ods', 'odp']:
            return "opendocument"
        
        # If raw_file_type doesn't match, try filename
        if filename:
            file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if file_ext in ['xlsx', 'xls']:
                return "excel"
            elif file_ext == 'pdf':
                return "pdf"
            elif file_ext in ['docx', 'doc']:
                return "word"
            elif file_ext in ['pptx', 'ppt']:
                return "powerpoint"
            elif file_ext == 'csv':
                return "csv"
            elif file_ext in ['txt', 'md']:
                return "text"
            elif file_ext == 'json':
                return "json"
            elif file_ext in ['odt', 'ods', 'odp']:
                return "opendocument"
        
        # If still no match, try MIME type
        if mime_type:
            if 'spreadsheet' in mime_type or 'excel' in mime_type:
                return "excel"
            elif 'pdf' in mime_type:
                return "pdf"
            elif 'word' in mime_type or 'document' in mime_type:
                return "word"
            elif 'presentation' in mime_type:
                return "powerpoint"
            elif 'csv' in mime_type:
                return "csv"
            elif 'json' in mime_type:
                return "json"
            elif 'text' in mime_type:
                return "text"
        
        # Default fallback
        return "document"
    
    def _initialize_llms(self):
        """Initialize language models with tools"""
        # Define tools
        @tool
        def generate_image_tool(prompt: str) -> dict:
            """
            Generates an image from a textual prompt using a text-to-image model.
            The image is returned as a base64 encoded string along with its MIME type.
            
            IMPORTANT USAGE INSTRUCTIONS:
            1. DIRECTLY INVOKE this tool when the user asks to "generate an image", "create a picture", "draw", etc.
            2. DO NOT output code or function calls as text - use the proper tool invocation mechanism
            3. DO NOT write "generate_image_tool(prompt='...')" as text in your response
            4. Keep the prompt SIMPLE and FAITHFUL to what the user requested
            
            Example good prompts:
            - "A cat wearing a hat"
            - "A mountain landscape with a lake"
            - "A simple house with a red roof"
            
            The system will automatically simplify overly complex prompts to match the user's intent.
            """
            return self._generate_image_from_hf_api(prompt)

        @tool
        def generate_video_tool(prompt: str) -> dict:
            """
            Generates a video from a textual prompt using a text-to-video model.
            The video is returned as a base64 encoded string along with its MIME type.
            Use this tool when the user explicitly asks to "generate a video", "create a video", "make a video animation", or similar.
            The 'prompt' argument for this tool should be a highly detailed and descriptive textual representation of the desired video.
            """
            return self._generate_video_from_hf_api(prompt)
        
        # Initialize Gemini 1.5 Flash LLM with tools
        try:
            self.gemini_flash_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.GOOGLE_API_KEY,
                temperature=0.7
            ).bind_tools([generate_image_tool, generate_video_tool])
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini 1.5 Flash: {e}")

        # Initialize Gemini 2.5 Pro LLM with tools
        try:
            self.gemini_pro_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=self.GOOGLE_API_KEY,
                temperature=0.7
            ).bind_tools([generate_image_tool, generate_video_tool])
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini 2.5 Pro: {e}")
        
        # Initialize ChatGPT model with tools
        try:
            self.chatgpt_llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=self.CHATGPT_TOKEN,
                temperature=0.7
            ).bind_tools([generate_image_tool, generate_video_tool])
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChatGPT: {e}")

        # Initialize DeepSeek model through OpenRouter with tools
        try:
            self.deepseek_llm = ChatOpenAI(
                model="deepseek/deepseek-chat-v3-0324",
                api_key=self.OPEN_ROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.7
            ).bind_tools([generate_image_tool, generate_video_tool])
        except Exception as e:
            logger.error(f"❌ Failed to initialize DeepSeek: {e}")

        # Initialize Mistral model through OpenRouter with tools
        try:
            self.mistral_llm = ChatOpenAI(
                model="mistralai/mistral-nemo",
                api_key=self.OPEN_ROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.7
            ).bind_tools([generate_image_tool, generate_video_tool])
        except Exception as e:
            logger.error(f"❌ Failed to initialize Mistral: {e}")

        # Initialize Gemini 2.0 Flash Exp with Google API key
        try:
            self.gemini_flash_exp_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=self.GOOGLE_API_KEY,
                temperature=0.7
            ).bind_tools([generate_image_tool, generate_video_tool])
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini 2.0 Flash Exp: {e}")
    
    def _simplify_image_prompt(self, original_prompt: str, original_question: str = None) -> str:
        """
        Simplifies an image generation prompt to be more faithful to the user's original request.
        This prevents the LLM from creating overly complex, embellished prompts.
        
        Args:
            original_prompt: The prompt generated by the LLM
            original_question: The user's original question/request
            
        Returns:
            A simplified prompt that's more faithful to the original request
        """
        # If we don't have the original question, use the original prompt
        if not original_question:
            return original_prompt
            
        # Clean up the user question - remove "draw", "generate", etc. but keep the subject
        clean_question = re.sub(r'^(draw|create|generate|make|show)\s+(an?|the)?\s+', '', original_question.lower())
        clean_question = re.sub(r'^(picture|image|illustration|photo|photograph)\s+of\s+', '', clean_question)
        clean_question = clean_question.strip()
        
        # For extremely simple prompts, add minimal style guidance
        if len(clean_question.split()) <= 5:
            return f"A simple, clear illustration of {clean_question}, on a plain background"
        
        # For longer prompts, preserve more of the user's intent but still keep it simple
        return f"A clean illustration of {clean_question}"
    
    def _generate_image_from_hf_api(self, prompt_text: str) -> dict:
        """Internal function to call the Hugging Face text-to-image API"""
        logger.info(f"🎨 IMAGE GENERATION PROMPT TO HUGGING FACE: '{prompt_text}'") # Log prompt to HF
        
        payload = {
            "inputs": prompt_text,
            "parameters": {
                "negative_prompt": "blurry, low quality, bad anatomy, deformed, ugly, extra limbs, watermark, text, signature, cartoon, illustration, painting, sketch, drawing, abstract, digital art, low resolution, dull, dark, muted colors, poor lighting, underexposed, overexposed",
                "guidance_scale": 7.5, 
                "num_inference_steps": 50,
                "height": 1024, 
                "width": 1024  
            },
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }

        try:
            hf_start = time.time()
            response = requests.post(self.HF_SD_MODEL_URL, headers=self.HF_HEADERS, json=payload, timeout=120)
            response.raise_for_status()
            hf_end = time.time()
            
            image_bytes = response.content
            image_mime_type = response.headers.get("Content-Type", "image/jpeg")
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Log successful image generation with time taken
            logger.info(f"✅ IMAGE GENERATED SUCCESSFULLY BY HUGGING FACE - Size: {len(image_bytes)} bytes | Time: {hf_end - hf_start:.2f}s")
            
            return {
                "status": "success",
                "image_data_base64": base64_image,
                "image_mime_type": image_mime_type,
                "prompt_used": prompt_text
            }

        except requests.exceptions.HTTPError as e:
            error_msg = f"Hugging Face API HTTP Error: {e.response.status_code} - {e.response.text}"
            logger.error(f"❌ IMAGE GENERATION FAILED (HUGGING FACE): {error_msg}")
            return {"status": "error", "message": error_msg}
        except requests.exceptions.Timeout:
            error_msg = "Hugging Face API request timed out. Model might be loading or busy."
            logger.error(f"⏰ IMAGE GENERATION TIMEOUT (HUGGING FACE): {error_msg}")
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"An unexpected error occurred during image processing: {e}"
            logger.error(f"💥 IMAGE GENERATION ERROR (HUGGING FACE): {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _generate_video_from_hf_api(self, prompt_text: str) -> dict:
        """Internal function to call the Hugging Face text-to-video API"""
        logger.info(f"🎬 VIDEO GENERATION PROMPT TO HUGGING FACE: '{prompt_text}'") # Log prompt to HF
        
        payload = {
            "inputs": prompt_text,
            "parameters": {
                "num_frames": 16,
                "fps": 8,
            },
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }

        try:
            if "YOUR_TEXT_TO_VIDEO_MODEL_ENDPOINT" in self.HF_VIDEO_MODEL_URL:
                raise ValueError("HF_VIDEO_MODEL_URL is a placeholder. Please replace it with your actual deployed Hugging Face Inference Endpoint for a text-to-video model.")

            hf_start = time.time()
            response = requests.post(self.HF_VIDEO_MODEL_URL, headers=self.HF_HEADERS, json=payload, timeout=300)
            response.raise_for_status()
            hf_end = time.time()

            video_bytes = response.content
            video_mime_type = response.headers.get("Content-Type", "video/mp4")
            base64_video = base64.b64encode(video_bytes).decode('utf-8')
            
            # Log successful video generation with time taken
            logger.info(f"✅ VIDEO GENERATED SUCCESSFULLY BY HUGGING FACE - Size: {len(video_bytes)} bytes | Time: {hf_end - hf_start:.2f}s")
            
            return {
                "status": "success",
                "video_data_base64": base64_video,
                "video_mime_type": video_mime_type,
                "prompt_used": prompt_text
            }

        except ValueError as e:
            error_msg = f"Video tool configuration error: {e}"
            logger.error(f"❌ VIDEO TOOL ERROR: {error_msg}")
            return {"status": "error", "message": error_msg}
        except requests.exceptions.HTTPError as e:
            error_msg = f"Hugging Face Video API HTTP Error: {e.response.status_code} - {e.response.text}"
            logger.error(f"❌ VIDEO TOOL ERROR: {error_msg}")
            return {"status": "error", "message": error_msg}
        except requests.exceptions.Timeout:
            error_msg = "Hugging Face Video API request timed out. Video generation takes longer than images."
            logger.error(f"⏰ VIDEO TOOL ERROR: {error_msg}")
            return {"status": "error", "message": error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"An unexpected network error occurred with Hugging Face Video API: {e}"
            logger.error(f"❌ VIDEO TOOL ERROR: {error_msg}")
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"An unexpected error occurred during video processing: {e}"
            logger.error(f"💥 VIDEO TOOL ERROR: {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _process_file_with_rag(self, file_bytes: bytes, filename: str, mime_type: str, user_email: str, thread_id: int, user_id: int = None, db: Session = None) -> Dict[str, Any]:
        """Process file using RAG - convert to vectors and store in ChromaDB for specific thread"""
        try:
            # Add file to vector store with database session and thread_id
            result = self.rag_service.add_file_to_vectorstore(file_bytes, filename, mime_type, user_email, thread_id, user_id, db)
            
            if result["status"] == "success":
                raw_file_type = result.get("file_type", "file")
                # Map RAG file types to UI-friendly types
                file_type = self._map_file_type_for_ui(raw_file_type, filename, mime_type)
                logger.info(f"✅ {raw_file_type.upper()} processed with RAG - {user_email} | Thread: {thread_id} | Filename: {filename} | Chunks: {result['chunks_count']}")
                return {
                    "rag_result": result,
                    "filename": filename,
                    "file_type": file_type,
                    "mime_type": mime_type,
                    "processed_with_rag": True
                }
            elif result["status"] == "exists":
                raw_file_type = result.get("file_type", "file")
                # Map RAG file types to UI-friendly types
                file_type = self._map_file_type_for_ui(raw_file_type, filename, mime_type)
                logger.info(f"📋 {raw_file_type.upper()} already in RAG store - {user_email} | Thread: {thread_id} | Filename: {filename}")
                return {
                    "rag_result": result,
                    "filename": filename,
                    "file_type": file_type,
                    "mime_type": mime_type,
                    "processed_with_rag": True
                }
            else:
                logger.error(f"❌ RAG processing failed - {user_email} | Filename: {filename} | Error: {result.get('message', 'Unknown error')}")
                return {
                    "error": result.get("message", "Failed to process file with RAG"),
                    "filename": filename,
                    "mime_type": mime_type,
                    "processed_with_rag": False
                }
                
        except Exception as e:
            logger.error(f"❌ RAG processing error - {user_email} | Filename: {filename} | Error: {e}")
            return {
                "error": f"Error processing file with RAG: {str(e)}",
                "filename": filename,
                "mime_type": mime_type,
                "processed_with_rag": False
            }
    
    # Keep backward compatibility for PDF method
    def _process_pdf_with_rag(self, pdf_bytes: bytes, filename: str, user_email: str, thread_id: int, user_id: int = None, db: Session = None) -> Dict[str, Any]:
        """Process PDF using RAG - backward compatibility wrapper"""
        return self._process_file_with_rag(pdf_bytes, filename, 'application/pdf', user_email, thread_id, user_id, db)
    
    async def _store_invoice_in_rag(self, validation_results: Dict[str, Any], 
                                     filename: str, user_email: str, thread_id: int, user_id: int = None, db: Session = None) -> Dict[str, Any]:
        """Store only extracted invoice information in ChromaDB vector store (without validation rules)"""
        try:
            # Get extracted data from validation results
            invoice_details = validation_results['extracted_data']
            
            # Create a clean text representation with only extracted information
            invoice_text = f"""
INVOICE DOCUMENT: {filename}

EXTRACTED INFORMATION:
- Invoice Number: {invoice_details.get('invoice_number', 'Not found')}
- Invoice Date: {invoice_details.get('invoice_date', 'Not found')}
- Vendor/Store Name: {invoice_details.get('vendor_name', 'Not found')}
- Total Amount: {invoice_details.get('total_amount', 'Not found')}
- Tax Amount: {invoice_details.get('tax_amount', 'Not found')}
- Buyer Details: {invoice_details.get('buyer_details', 'Not found')}

LINE ITEMS:
"""
            
            line_items = invoice_details.get('line_items', [])
            for i, item in enumerate(line_items, 1):
                invoice_text += f"{i}. {item}\n"
            
            if not line_items:
                invoice_text += "No line items detected\n"
            
            invoice_text += f"""
EXTRACTED TEXT CONTENT:
{invoice_details.get('raw_ocr_text', 'No OCR text available')}
"""
            
            # Store in RAG using the text representation
            invoice_bytes = invoice_text.encode('utf-8')
            result = self.rag_service.add_file_to_vectorstore(
                invoice_bytes, 
                f"invoice_{filename}", 
                'text/plain', 
                user_email, 
                thread_id, 
                user_id, 
                db
            )
            
            if result["status"] == "success":
                logger.info(f"✅ INVOICE stored in ChromaDB - {user_email} | Thread: {thread_id} | Filename: {filename}")
                return result
            else:
                logger.error(f"❌ Failed to store invoice in ChromaDB: {result.get('message', 'Unknown error')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Error storing invoice in RAG: {e}")
            return {"status": "error", "message": f"Failed to store invoice: {str(e)}"}
    
    async def process_file_upload(self, file: UploadFile, user_email: str, thread_id: int, user_id: int = None, db: Session = None) -> Dict[str, Any]:
        """Process uploaded file and return metadata"""
        file_bytes = await file.read()
        file_mimetype = file.content_type
        
        logger.info(f"⬆️ FILE UPLOADED - {user_email} | Thread: {thread_id} | Filename: {file.filename} | Mime Type: {file_mimetype}")
        
        # Special handling for images - check if it's an invoice
        if file_mimetype and file_mimetype.startswith("image/"):
            file_content_base64 = base64.b64encode(file_bytes).decode('utf-8')
            
            # Check if the image is an invoice
            is_invoice = self.invoice_service.classify_image_as_invoice(file_content_base64)
            
            if is_invoice:
                logger.info(f"📄 INVOICE DETECTED - {user_email} | Will process with LLM's built-in OCR")
                
                # Return simplified invoice result - let LLM handle OCR and validation
                return {
                    "file_content_base64": file_content_base64,
                    "file_mimetype": file_mimetype,
                    "filename": file.filename,
                    "processed_with_rag": False,  # Don't store in RAG, let LLM process directly
                    "is_invoice": True,
                    "is_media": True
                }
            else:
                # Regular image - not an invoice
                return {
                    "file_content_base64": file_content_base64,
                    "file_mimetype": file_mimetype,
                    "filename": file.filename,
                    "processed_with_rag": False,
                    "is_invoice": False,
                    "is_media": True
                }
        
        # Define document types that should be processed with RAG
        document_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
            "application/msword",  # doc
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx
            "application/vnd.ms-powerpoint",  # ppt
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
            "application/vnd.ms-excel",  # xls
            "text/csv",
            "application/csv",
            "text/plain",  # txt
            "application/json",
            "text/markdown",  # md
            "application/vnd.oasis.opendocument.text",  # odt
            "application/vnd.oasis.opendocument.spreadsheet",  # ods
            "application/vnd.oasis.opendocument.presentation"  # odp
        ]
        
        # Check file extension for document types
        file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        document_extensions = ['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', 'csv', 'txt', 'json', 'md', 'odt', 'ods', 'odp']
        
        # Process documents with RAG
        if file_mimetype in document_types or file_ext in document_extensions:
            rag_result = self._process_file_with_rag(file_bytes, file.filename, file_mimetype, user_email, thread_id, user_id, db)
            return rag_result
        else:
            # For videos and other media files, store as base64 (keep existing behavior)
            file_content_base64 = base64.b64encode(file_bytes).decode('utf-8')
            return {
                "file_content_base64": file_content_base64,
                "file_mimetype": file_mimetype,
                "filename": file.filename,
                "processed_with_rag": False,
                "is_media": True  # Flag to indicate this is media content
            }
    
    def save_user_message(self, message: str, thread_id: int, file_data: Optional[Dict[str, Any]], db: Session, user_email: str) -> Chat:
        """Save user message to database"""
        # Only store image data for actual images, NOT for PDFs
        image_data = None
        image_mime_type = None
        video_data = None
        video_mime_type = None
        filename = None
        file_type = None
        
        if file_data:
            filename = file_data.get("filename")
            file_type = file_data.get("file_type")  # Get file type from file_data
            
            # If file_type is not present in file_data, determine it from filename or mime_type
            if not file_type and filename:
                file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
                if file_ext in ['pdf']:
                    file_type = "pdf"
                elif file_ext in ['xlsx', 'xls']:
                    file_type = "excel"
                elif file_ext in ['docx', 'doc']:
                    file_type = "word"
                elif file_ext in ['pptx', 'ppt']:
                    file_type = "powerpoint"
                elif file_ext in ['csv']:
                    file_type = "csv"
                elif file_ext in ['txt', 'md']:
                    file_type = "text"
                elif file_ext in ['json']:
                    file_type = "json"
                elif file_ext in ['odt', 'ods', 'odp']:
                    file_type = "opendocument"
                elif file_data.get("file_mimetype", "").startswith("image/"):
                    file_type = "image"
                elif file_data.get("file_mimetype", "").startswith("video/"):
                    file_type = "video"
                else:
                    file_type = "document"  # fallback
            
            if file_data.get("processed_with_rag") and not (file_data.get("is_invoice") and file_data.get("file_mimetype", "").startswith("image/")):
                # For files processed with RAG (PDF, Word, Excel, etc.), don't store media data
                # But for invoice images, we still want to show the image in UI
                # Keep the determined file_type for RAG-processed documents
                pass
            elif file_data.get("file_mimetype", "").startswith("image/"):
                # For images (including invoices) - store both the base64 data AND the MIME type
                image_data = file_data["file_content_base64"]
                image_mime_type = file_data["file_mimetype"]
                file_type = "image"  # Override for images
            elif file_data.get("file_mimetype", "").startswith("video/"):
                # For videos - store video data
                video_data = file_data.get("file_content_base64")
                video_mime_type = file_data["file_mimetype"]
                file_type = "video"  # Override for videos
        
        user_chat = Chat(
            sender="user",
            message=message,
            thread_id=thread_id,
            # CORRECTED TYPO HERE: image_data_base664 -> image_data_base64
            image_data_base64=image_data,        # Column exists - stores base64 for images only
            image_mime_type=image_mime_type,     # Column exists - stores MIME type for images only
            video_data_base64=video_data,        # Column exists - stores base64 for videos only
            video_mime_type=video_mime_type,     # Column exists - stores MIME type for videos only
            filename=filename,                   # Column exists - stores filename for any file type
            file_type=file_type                  # Column exists - stores file type
        )
        db.add(user_chat)
        db.commit()
        db.refresh(user_chat)
        
        logger.info(f"💾 USER MESSAGE SAVED - {user_email} | Thread ID: {thread_id} | Message: '{message[:100]}...' | Filename: {filename} | File Type: {file_type}")
        return user_chat
    
    def save_bot_message(self, message: str, thread_id: int, image_data: Optional[str], 
                             image_mime_type: Optional[str], video_data: Optional[str], 
                             video_mime_type: Optional[str], db: Session, user_email: str) -> Chat:
        """Save bot message to database"""
        try:
            # Determine file_type based on what data is present
            file_type = None
            if image_data and image_mime_type:
                file_type = "image"
            elif video_data and video_mime_type:
                file_type = "video"
            
            bot_chat = Chat(
                sender="bot",
                message=message,
                thread_id=thread_id,
                image_data_base64=image_data,
                image_mime_type=image_mime_type,
                video_data_base64=video_data,
                video_mime_type=video_mime_type,
                file_type=file_type
            )
        except Exception as e:
            bot_chat = Chat(
                sender="bot",
                message=message,
                thread_id=thread_id
            )

        db.add(bot_chat)
        db.commit()
        db.refresh(bot_chat)
        
        # Fix the logging with proper log_content initialization
        log_content = f"💾 BOT MESSAGE SAVED - {user_email} | Thread ID: {thread_id}"
        if message:
            log_content += f" | Message: '{message[:100]}...'"
        if image_data:
            log_content += " | Image: Present"
        if video_data:
            log_content += " | Video: Present"
        
        logger.info(log_content)
        return bot_chat
    
    def get_recent_chat_history(self, thread_id: int, limit: int, db: Session, user_email: str) -> List[Chat]:
        """Get recent chat history for context"""
        recent_chats_db = (
            db.query(Chat)
            .filter(Chat.thread_id == thread_id)
            .order_by(Chat.timestamp.desc())
            .limit(limit)
            .all()
        )
        
        recent_chats = recent_chats_db[::-1]  # Reverse to get chronological order
        return recent_chats
    
    def build_langchain_messages(self, recent_chats: List[Chat], current_question: str, 
                                 file_data: Optional[Dict[str, Any]], user_email: str, thread_id: int) -> List:
        """Build LangChain message format for LLM"""
        langchain_messages = []
        
        # Add system prompt
        langchain_messages.append(SystemMessage(content=self.SYSTEM_PROMPT))
        
        # Add conversation history
        if recent_chats:
            langchain_messages.append(SystemMessage(content="Here is the recent conversation history for context:"))
            
            for chat_entry in recent_chats:
                if chat_entry.sender == "user":
                    # Only include image data for actual images, NOT PDFs
                    if (chat_entry.image_data_base64 and 
                        hasattr(chat_entry, 'image_mime_type') and 
                        chat_entry.image_mime_type and 
                        chat_entry.image_mime_type.startswith('image/')):
                        langchain_messages.append(HumanMessage(content=[
                            {"type": "text", "text": chat_entry.message},
                            {"type": "image_url", "image_url": {"url": f"data:{chat_entry.image_mime_type};base64,{chat_entry.image_data_base64}"}}
                        ]))
                    else:
                        langchain_messages.append(HumanMessage(content=chat_entry.message))
                elif chat_entry.sender == "bot":
                    # For bot messages, if they contain image/video data, include it
                    if chat_entry.image_data_base64 and chat_entry.image_mime_type:
                        langchain_messages.append(AIMessage(content=[
                            {"type": "text", "text": chat_entry.message},
                            {"type": "image_url", "image_url": {"url": f"data:{chat_entry.image_mime_type};base64,{chat_entry.image_data_base64}"}}
                        ]))
                    elif chat_entry.video_data_base64 and chat_entry.video_mime_type:
                        # Langchain does not directly support video_url in messages,
                        # so we'll just include the text and note the video.
                        # The actual video will be sent via API response outside this message chain.
                        langchain_messages.append(AIMessage(content=chat_entry.message + " (Video content was generated)"))
                    else:
                        langchain_messages.append(AIMessage(content=chat_entry.message))
        
        # Add context instruction (this should still be present for general conversation flow)
        langchain_messages.append(SystemMessage(content="Please carefully use the conversation history provided above to answer the user's current query, especially for follow-up questions that implicitly refer to a past topic or entity. Your goal is to maintain a coherent and context-aware conversation."))
        
        # Add current user question with RAG enhancement
        question_to_llm = current_question
        
        # --- IMPORTANT MODIFICATION STARTS HERE ---
        # Handle invoice processing as the HIGHEST PRIORITY and most explicit instruction
        if file_data and file_data.get("is_invoice"):
            logger.info(f"📄 Processing invoice image with LLM for direct analysis - {user_email}")
            
            # This prompt now explicitly demands the structured output.
            # It's tailored for the LLM to understand it's a direct task, not a conversation starter.
            invoice_analysis_instruction = f"""
            INVOICE ANALYSIS TASK: Please analyze the provided invoice image and extract all relevant information.
            Your response MUST strictly follow the REQUIRED OUTPUT FORMAT below. DO NOT add any introductory or conversational text. Simply provide the structured report.

            User's specific request for this invoice: {current_question if current_question else 'Perform a complete analysis and validation of this invoice.'}

            **REQUIRED OUTPUT FORMAT:**

            Document Type: Invoice

            Extracted Fields:
            - Invoice Number: [extracted number or "Not detected"]
            - Invoice Date: [extracted date or "Not detected"]
            - Vendor Name: [extracted vendor name or "Not detected"]
            - Buyer Name: [extracted buyer name or "Not detected"]
            - Items: [List items in the format '[Quantity]x [Description] @ $[Unit Price]' or "Not detected" (e.g., [2x Widget A @ $10, 1x Widget B @ $20])]
            - Total Amount: [extracted total or "Not detected"]
            - Taxes: [extracted tax amount or "Not detected"]
            - Payment Due Date: [extracted due date or "Not detected"]

            Validation Results:
            - Invoice number present and clear: [pass/fail] — [Explanation, e.g., 'Found as INV-12345']
            - Invoice date present and formatted: [pass/fail] — [Explanation, e.g., 'Detected 12/06/2025 (DD/MM/YYYY)']
            - Vendor name present and clear: [pass/fail] — [Explanation, e.g., 'ABC Supplies']
            - At least one item/service listed: [pass/fail] — [Explanation, e.g., '2 items detected']
            - Total amount present and numeric: [pass/fail] — [Explanation, e.g., '$40']
            - Image quality sufficient: [pass/fail] — [Explanation, e.g., 'No significant blur/cropping']

            Score: [X]/6 ([X]%)

            **Strict adherence to this format is crucial.** Ensure all parts of the format are present.
            """
            
            langchain_messages.append(HumanMessage(content=[
                {"type": "text", "text": invoice_analysis_instruction},
                {"type": "image_url", "image_url": {"url": f"data:{file_data['file_mimetype']};base64,{file_data['file_content_base64']}"}}
            ]))
            
            # Crucially, return here. This stops further processing of general RAG or other file types
            # for this specific message turn, ensuring the LLM only focuses on the invoice task.
            return langchain_messages 

        # Handle regular images (not invoices) - MUST come before general question handling
        elif file_data and file_data.get("file_mimetype", "").startswith("image/") and not file_data.get("is_invoice"):
            # For regular images, include the image data directly without searching RAG
            if not current_question or not current_question.strip():
                question_to_llm = "Please analyze this image."
            else:
                question_to_llm = current_question
                
            logger.info(f"🖼️ Processing regular image with LLM - {user_email} | Filename: {file_data.get('filename')}")
            
            langchain_messages.append(HumanMessage(content=[
                {"type": "text", "text": question_to_llm},
                {"type": "image_url", "image_url": {"url": f"data:{file_data['file_mimetype']};base64,{file_data['file_content_base64']}"}}
            ]))
            
            # Return here to prevent RAG search for image analysis
            return langchain_messages
        # --- IMPORTANT MODIFICATION ENDS HERE ---

        # Handle new document upload case (PDF, Excel, Word, etc.) - prioritize the currently uploaded file
        elif file_data and file_data.get("processed_with_rag"):
            if current_question:
                filename = file_data.get("filename")
                file_type = file_data.get("file_mimetype", "").split("/")[-1].upper()  # Get file type for logging
                logger.info(f"🎯 RAG: Searching in newly uploaded file '{filename}' - {user_email}")
                
                file_specific_chunks = self.rag_service.search_in_specific_file(
                    query=current_question,
                    filename=filename,
                    user_email=user_email,
                    thread_id=thread_id,
                    top_k=3
                )
                
                if file_specific_chunks:
                    rag_context = "\n\n".join([
                        f"From the uploaded file '{chunk['filename']}' (similarity: {chunk['similarity_score']:.2f}):\n{chunk['content']}"
                        for chunk in file_specific_chunks
                    ])
                    question_to_llm = f"{current_question}\n\nContent from the uploaded file:\n{rag_context}"
                    logger.info(f"🎯 RAG: Found {len(file_specific_chunks)} chunks in uploaded file '{filename}' - {user_email}")
                else:
                    relevant_chunks = self.rag_service.search_relevant_content(
                        query=current_question, 
                        user_email=user_email,
                        thread_id=thread_id,
                        top_k=3
                    )
                    
                    if relevant_chunks:
                        rag_context = "\n\n".join([
                            f"From '{chunk['filename']}' (similarity: {chunk['similarity_score']:.2f}):\n{chunk['content']}"
                            for chunk in relevant_chunks
                        ])
                        question_to_llm = f"{current_question}\n\nRelevant context from your documents (note: no relevant content found in the uploaded file '{filename}'):\n{rag_context}"
                        logger.info(f"🔍 RAG: Fallback search found {len(relevant_chunks)} chunks from other files - {user_email}")
                    else:
                        question_to_llm = f"{current_question}\n\n(Note: File '{filename}' was uploaded but no relevant content found for this query.)"
                        logger.info(f"🔍 RAG: No relevant content found for uploaded file '{filename}' - {user_email}")
            else:
                question_to_llm = f"File '{file_data['filename']}' has been successfully processed and stored in the knowledge base."
            
            langchain_messages.append(HumanMessage(content=question_to_llm))
            
        elif current_question:
            # For regular questions (no new file upload), search across all user's documents
            # Detect if this is a numerical/ranking query and increase search scope
            is_numerical_query = any(keyword in current_question.lower() for keyword in [
                'rank', 'position', 'number', 'score', 'rating', 'place', 'order',
                '1st', '2nd', '3rd', 'th rank', 'st rank', 'nd rank', 'rd rank'
            ]) or any(char.isdigit() for char in current_question)
            
            search_top_k = 8 if is_numerical_query else 3  # Increase scope for numerical queries
            
            # Enhance query for better numerical searches
            enhanced_query = current_question
            if is_numerical_query:
                # Extract numbers and ranking terms to create multiple search variations
                import re
                numbers = re.findall(r'\d+', current_question)
                if numbers:
                    # Create enhanced query with variations
                    enhanced_query = f"{current_question} rank ranking position number {' '.join(numbers)}"
                    logger.info(f"🔢 NUMERICAL QUERY DETECTED - Enhanced search query - {user_email}")
            
            relevant_chunks = self.rag_service.search_relevant_content(
                query=enhanced_query, 
                user_email=user_email,
                thread_id=thread_id,
                top_k=search_top_k
            )
            
            if relevant_chunks:
                rag_context = "\n\n".join([
                    f"From '{chunk['filename']}' (similarity: {chunk['similarity_score']:.2f}):\n{chunk['content']}"
                    for chunk in relevant_chunks
                ])
                question_to_llm = f"{current_question}\n\nRelevant context from your uploaded documents:\n{rag_context}"
                logger.info(f"🔍 RAG: Found {len(relevant_chunks)} relevant chunks for query - {user_email}")
            else:
                # No relevant content found - try a broader search for numerical queries
                if is_numerical_query:
                    logger.info(f"🔍 RAG: No chunks found, trying broader search for numerical query - {user_email}")
                    # Try searching with just the numbers and common ranking terms
                    numbers = re.findall(r'\d+', current_question)
                    if numbers:
                        fallback_query = f"rank ranking position {' '.join(numbers)} country countries"
                        relevant_chunks = self.rag_service.search_relevant_content(
                            query=fallback_query,
                            user_email=user_email,
                            thread_id=thread_id,
                            top_k=10  # Cast wider net
                        )
                        if relevant_chunks:
                            rag_context = "\n\n".join([
                                f"From '{chunk['filename']}' (similarity: {chunk['similarity_score']:.2f}):\n{chunk['content']}"
                                for chunk in relevant_chunks
                            ])
                            question_to_llm = f"{current_question}\n\nBroader context from your uploaded documents:\n{rag_context}"
                            logger.info(f"🔍 RAG: Fallback search found {len(relevant_chunks)} chunks - {user_email}")
                        else:
                            logger.info(f"🔍 RAG: No relevant chunks found even with fallback search - {user_email}")
                else:
                    logger.info(f"🔍 RAG: No relevant chunks found for query - {user_email}")
            
            langchain_messages.append(HumanMessage(content=question_to_llm))
        
        # Handle error cases and other file types (ensure correct mimetype checks)
        elif file_data and not file_data.get("processed_with_rag") and file_data.get("error"):
            # File processing failed
            error_msg = file_data.get("error", "Unknown error occurred while processing file")
            question_to_llm = f"{current_question}\n\n(Error: {error_msg})" if current_question else f"Error processing file: {error_msg}"
            langchain_messages.append(HumanMessage(content=question_to_llm))
            
        elif file_data: # General file handling for non-RAG files (videos, etc. - images are handled above)
            question_to_llm = f"{current_question}\n\n(User provided a file: {file_data['filename']})." if current_question else f"User provided a file: {file_data['filename']}."
            if not question_to_llm or not question_to_llm.strip():
                question_to_llm = f"User provided a file: {file_data.get('filename', 'unknown file')}."
            langchain_messages.append(HumanMessage(content=question_to_llm))
        else: # No file uploaded, no question asked (shouldn't happen with current UI flow, but as a fallback)
            if not question_to_llm or not question_to_llm.strip():
                question_to_llm = "Hello! How can I help you today?"
            langchain_messages.append(HumanMessage(content=question_to_llm))
        
        return langchain_messages
    
    def _is_sql_question(self, question: str) -> bool:
        """Detect if a question is asking about database/SQL data"""
        # Direct database references - these are clear indicators
        direct_sql_keywords = [
            'dvdrental', 'database', 'table', 'customer', 'film', 'actor', 'rental', 'payment', 'store',
            'inventory', 'address', 'city', 'country', 'category', 'staff'
        ]
        
        # SQL-specific data questions - only when combined with database context
        sql_data_keywords = [
            'with id', 'id =', 'customer id', 'film id', 'actor id',
            'person with', 'movie with', 'film with',
            'customers', 'movies', 'films', 'actors', 'rentals', 'payments'
        ]
        
        question_lower = question.lower()
        
        # First check for direct database references
        for keyword in direct_sql_keywords:
            if keyword in question_lower:
                logger.info(f"🗃️ SQL QUESTION DETECTED - Direct keyword: '{keyword}'")
                return True
        
        # Check for SQL data keywords (these need database context)
        for keyword in sql_data_keywords:
            if keyword in question_lower:
                logger.info(f"🗃️ SQL QUESTION DETECTED - Data keyword: '{keyword}'")
                return True
        
        # Check for very specific SQL patterns (avoid generic 'who is', 'what are', etc.)
        sql_patterns = [
            r'customer.*name.*\w+',  # "customer name John"
            r'film.*title.*\w+',     # "film title something"
            r'actor.*\d+',           # "actor 123"
            r'rental.*date',         # "rental date"
            r'payment.*amount'       # "payment amount"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, question_lower):
                logger.info(f"🗃️ SQL QUESTION DETECTED - Pattern: '{pattern}'")
                return True
        
        logger.info(f"❌ NOT SQL QUESTION - No database context found")
        return False
    
    def _is_file_question(self, question: str) -> bool:
        """Detect if a question is asking about uploaded files/documents"""
        file_keywords = [
            'above file', 'this file', 'uploaded file', 'the file', 'document', 'pdf',
            'according to', 'based on', 'from the file', 'in the file', 'file shows',
            'document says', 'pdf contains', 'extract from', 'summarize',
            'overview of', 'content of', 'what does', 'analyze'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in file_keywords)
    
    def _is_trip_planning_question(self, question: str) -> bool:
        """Detect if a question is asking about trip planning or vacation planning"""
        trip_keywords = [
            'plan a trip', 'trip to', 'vacation to', 'visit', 'travel to', 'holiday to',
            'itinerary', 'places to visit', 'tourist attractions', 'sightseeing',
            'day trip', 'weekend trip', 'vacation plan', 'travel plan', 'tour',
            'explore', 'destinations', 'trip planner', 'travel guide',
            'where to go', 'what to see', 'travel recommendations',
            'honeymoon trip', 'family trip', 'solo trip', 'group trip',
            'business trip', 'adventure trip', 'road trip', 'backpacking',
            'city break', 'getaway', 'excursion', 'journey'
        ]
        
        question_lower = question.lower()
        
        # Check for trip-related keywords
        for keyword in trip_keywords:
            if keyword in question_lower:
                logger.info(f"🌴 TRIP PLANNING QUESTION DETECTED - Keyword: '{keyword}'")
                return True
        
        # Enhanced patterns for natural language travel queries
        location_patterns = [
            # Handle "want to go to [location]" patterns
            r'want\s+to\s+go\s+(?:to\s+)?[A-Za-z\s,]+',
            r'want\s+to\s+go\s+for\s+[A-Za-z\s,]+',
            # Handle "go to [location]" patterns
            r'go\s+(?:to\s+)?[A-Za-z\s,]+\s+(?:next|this|in|for|\d+)',
            r'go\s+for\s+[A-Za-z\s,]+\s+(?:next|this|in|for|\d+)',
            # Standard patterns
            r'to\s+[A-Za-z\s,]+\s+(?:next|this|in|for|\d+)',
            r'visit\s+[A-Za-z\s,]+\s+(?:next|this|in|for|\d+)',
            r'trip\s+to\s+[A-Za-z\s,]+',
            r'\d+[-\s]day.*trip',
            r'plan.*\d+.*day.*to',
            # New patterns for natural language
            r'give\s+me.*trip.*plan',
            r'trip.*plan.*accordingly',
            r'plan.*accordingly'
        ]
        
        for pattern in location_patterns:
            if re.search(pattern, question_lower):
                logger.info(f"🌴 TRIP PLANNING QUESTION DETECTED - Pattern: '{pattern}'")
                return True
        
        logger.info(f"❌ NOT TRIP PLANNING QUESTION - No travel context found")
        return False
    
    def _has_files_in_thread(self, thread_id: int, user_email: str) -> bool:
        """Check if there are any uploaded files in the current thread"""
        try:
            # Search for any documents in this thread
            docs = self.rag_service.collection.get(
                where={
                    "$and": [
                        {"thread_id": thread_id},
                        {"user_email": user_email}
                    ]
                },
                limit=1
            )
            return len(docs['ids']) > 0
        except Exception as e:
            logger.error(f"❌ Error checking thread files: {e}")
            return False

    async def _handle_web_search_question(self, question: str, user_email: str) -> Dict[str, Any]:
        """Handle web search questions using the Web Search service"""
        try:
            logger.info(f"🌐 WEB SEARCH QUESTION DETECTED - {user_email}")
            logger.info(f"    Question: '{question}'")
            
            # Perform web search
            search_results = self.web_search_service.search_web(question, max_results=3)
            
            if search_results["status"] == "success":
                logger.info(f"✅ WEB SEARCH SUCCESS - {user_email}")
                logger.info(f"    Results: {len(search_results['results'])} found")
                logger.info(f"    Search time: {search_results['search_time']:.2f}s")
                
                # Format results for LLM
                formatted_results = self.web_search_service.format_search_results_for_llm(search_results)
                
                # Create a prompt for the LLM to synthesize the search results
                synthesis_prompt = f"""Based on the following web search results, please provide a comprehensive and accurate answer to the user's question: "{question}"

{formatted_results}

Please synthesize this information into a clear, helpful response. Focus on the most relevant and recent information. If there are conflicting details, mention that. 

IMPORTANT: You MUST include source references in your response. At the end of your answer, add a "🌐 Web Resources:" section that lists ALL website URLs where this information was found. Format it EXACTLY like this:

🌐 Web Resources: https://example1.com, https://example2.com, https://example3.com

- Use the exact label "🌐 Web Resources:" 
- Present all URLs side-by-side, comma-separated on a single continuous line
- Do NOT use bullets, numbers, or line breaks between sources
- Do NOT list source names separately - only the URLs
- This ensures optimal side-by-side appearance and allows the UI to handle any wrapping

This format allows users to easily access and verify the sources of the information."""

                return {
                    "web_search_synthesis_prompt": synthesis_prompt,
                    "original_question": question,
                    "search_results": search_results,
                    "requires_llm_synthesis": True
                }
            else:
                logger.error(f"❌ WEB SEARCH ERROR - {user_email} | Error: {search_results.get('message', 'Unknown error')}")
                # Fall back to regular LLM processing
                return None
                
        except Exception as e:
            logger.error(f"❌ WEB SEARCH PROCESSING ERROR - {user_email} | Error: {e}")
            # Fall back to regular LLM processing
            return None
    
    async def _handle_sql_question(self, question: str, user_email: str) -> Dict[str, Any]:
        try:
            logger.info(f"🗃️ SQL QUESTION DETECTED - {user_email}")
            logger.info(f"    Question: '{question}'")
            
            # Use SQL service to answer the question
            result = self.sql_service.answer_natural_language_question(question)
            
            if result["status"] == "success":
                logger.info(f"✅ SQL QUERY SUCCESS - {user_email}")
                logger.info(f"    Query: {result['sql_query']}")
                logger.info(f"    Results: {result['row_count']} rows")
                
                # Return just the natural language answer (no SQL query shown to user)
                return {
                    "bot_message_content": result["natural_language_answer"],
                    "image_data_base64": None,
                    "image_mime_type": None,
                    "video_data_base64": None,
                    "video_mime_type": None,
                    "is_sql_response": True
                }
            else:
                logger.error(f"❌ SQL QUERY ERROR - {user_email} | Error: {result.get('message', 'Unknown error')}")
                # Fall back to regular LLM processing
                return None
                
        except Exception as e:
            logger.error(f"❌ SQL PROCESSING ERROR - {user_email} | Error: {e}")
            # Fall back to regular LLM processing
            return None
    
    async def _handle_trip_planning_question(self, question: str, user_email: str) -> Dict[str, Any]:
        """Handle trip planning questions using the Trip Planning Agent with intelligent auto-correction"""
        try:
            logger.info(f"🌴 TRIP PLANNING QUESTION DETECTED - {user_email}")
            logger.info(f"    Question: '{question}'")
            
            # Extract location from question for auto-correction preview
            location_match = re.search(r'(?:to|visit|trip to|go to|in|plan.*trip.*to)\s+([A-Za-z\s,]+?)(?:\s+(?:for|next|this|in|\d+)|\s*$)', question.lower())
            if location_match:
                raw_location = location_match.group(1).strip()
                logger.info(f"🎯 Extracted location: '{raw_location}'")
                
                # Preview auto-correction before full processing
                corrected_location, confidence = self.trip_planning_agent._smart_location_correction(raw_location)
                if confidence >= 0.8 and corrected_location.lower() != raw_location.lower():
                    logger.info(f"🔧 Auto-correction preview: '{raw_location}' → '{corrected_location}' (confidence: {confidence:.2f})")
            
            # Use Trip Planning Agent to generate the itinerary
            try:
                itinerary = self.trip_planning_agent.main(question)
                
                if itinerary:
                    logger.info(f"✅ TRIP PLANNING SUCCESS - {user_email}")
                    logger.info(f"    Generated itinerary length: {len(itinerary)} characters")
                    
                    return {
                        "bot_message_content": itinerary,
                        "image_data_base64": None,
                        "image_mime_type": None,
                        "video_data_base64": None,
                        "video_mime_type": None,
                        "is_trip_planning_response": True
                    }
                else:
                    logger.error(f"❌ TRIP PLANNING ERROR - {user_email} | Empty itinerary returned")
                    return None
                    
            except Exception as trip_error:
                logger.error(f"❌ TRIP PLANNING AGENT ERROR - {user_email} | Error: {trip_error}")
                # Return a fallback message
                return {
                    "bot_message_content": f"I apologize, but I encountered an issue while planning your trip. The error was: {str(trip_error)}. Please try again with a different query or check if all required services are available.",
                    "image_data_base64": None,
                    "image_mime_type": None,
                    "video_data_base64": None,
                    "video_mime_type": None,
                    "is_trip_planning_response": True
                }
                
        except Exception as e:
            logger.error(f"❌ TRIP PLANNING PROCESSING ERROR - {user_email} | Error: {e}")
            # Fall back to regular LLM processing
            return None
    
    async def process_llm_response(self, langchain_messages: List, model: str, user_email: str, thread_id: int = None, original_question: str = None) -> Dict[str, Any]:
        """Process LLM response and handle tool calls"""
        
        # Initialize web search info
        websearch_info = None
        
        # Add debug logging for web search detection
        if original_question:
            is_web_search = self.web_search_service.is_web_search_query(original_question)
            is_file_question = thread_id and self._is_file_question(original_question) and self._has_files_in_thread(thread_id, user_email)
            is_sql_question = self._is_sql_question(original_question)
            is_trip_planning = self._is_trip_planning_question(original_question)
            
            logger.info(f"🔍 QUERY ANALYSIS - {user_email}")
            logger.info(f"    Question: '{original_question}'")
            logger.info(f"    Is Web Search Query: {is_web_search}")
            logger.info(f"    Is File Question: {is_file_question}")
            logger.info(f"    Is SQL Question: {is_sql_question}")
            logger.info(f"    Is Trip Planning Question: {is_trip_planning}")
        
        # First priority: Check if this is a question about uploaded files
        # The logic for invoice processing has been moved to build_langchain_messages
        # so this part primarily focuses on RAG for other documents or general file questions.
        if original_question and thread_id and self._is_file_question(original_question):
            if self._has_files_in_thread(thread_id, user_email):
                logger.info(f"📄 FILE QUESTION DETECTED - {user_email} | Thread: {thread_id}")
                logger.info(f"    Question: '{original_question}'")
                # Continue with normal RAG processing (files will be searched by langchain_messages already contains RAG context)
        
        # Second priority: Check if this is a web search question (only if not about files or SQL)
        elif original_question and self.web_search_service.is_web_search_query(original_question):
            # Only route to web search if it's not about files or database queries
            if not (thread_id and self._is_file_question(original_question) and self._has_files_in_thread(thread_id, user_email)) and not self._is_sql_question(original_question):
                logger.info(f"🌐 WEB SEARCH ROUTING ENABLED - {user_email}")
                web_search_result = await self._handle_web_search_question(original_question, user_email)
                if web_search_result and web_search_result.get("requires_llm_synthesis"):
                    # Store web search info for frontend
                    websearch_info = {
                        "search_performed": True,
                        "original_question": web_search_result["original_question"],
                        "results_count": len(web_search_result["search_results"]["results"]),
                        "search_time": web_search_result["search_results"]["search_time"]
                    }
                    
                    # Replace the original question with the synthesis prompt
                    synthesis_prompt = web_search_result["web_search_synthesis_prompt"]
                    # Update the langchain messages with the web search context
                    if langchain_messages and isinstance(langchain_messages[-1], HumanMessage):
                        langchain_messages[-1] = HumanMessage(content=synthesis_prompt)
                    # Continue with LLM processing using the enriched prompt
                    logger.info(f"🔄 WEB SEARCH -> LLM SYNTHESIS - {user_email}")
                else:
                    logger.warning(f"⚠️ WEB SEARCH FAILED - Falling back to direct LLM - {user_email}")
            else:
                logger.info(f"🚫 WEB SEARCH BLOCKED - File/SQL question detected - {user_email}")
            
        # Third priority: Check if this is a SQL question (only if not about files or web search)
        elif original_question and self._is_sql_question(original_question):
            # Only route to SQL if it's not about files or web search
            if not (thread_id and self._is_file_question(original_question) and self._has_files_in_thread(thread_id, user_email)) and not self.web_search_service.is_web_search_query(original_question):
                sql_result = await self._handle_sql_question(original_question, user_email)
                if sql_result:
                    return sql_result
        
        # Fourth priority: Check if this is a trip planning question (only if not about files, web search, or SQL)
        elif original_question and self._is_trip_planning_question(original_question):
            # Only route to trip planning if it's not about files, web search, or SQL
            if not (thread_id and self._is_file_question(original_question) and self._has_files_in_thread(thread_id, user_email)) and not self.web_search_service.is_web_search_query(original_question) and not self._is_sql_question(original_question):
                trip_result = await self._handle_trip_planning_question(original_question, user_email)
                if trip_result:
                    return trip_result
        
        # Select model
        current_llm = None
        if model == "gemini-1.5-flash":
            current_llm = self.gemini_flash_llm
            model_name_for_log = "Gemini 1.5 Flash"
        elif model == "gemini-2.5-pro":
            current_llm = self.gemini_pro_llm
            model_name_for_log = "Gemini 2.5 Pro"
        elif model == "gemini-2.0-flash-exp":
            current_llm = self.gemini_flash_exp_llm
            model_name_for_log = "Gemini 2.0 Flash Exp"
        elif model == "chatgpt":
            current_llm = self.chatgpt_llm
            model_name_for_log = "ChatGPT (gpt-4o-mini)"
        elif model == "deepseek-v3":
            current_llm = self.deepseek_llm
            model_name_for_log = "DeepSeek V3"
        elif model == "mistral-nemo":
            current_llm = self.mistral_llm
            model_name_for_log = "Mistral Nemo"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        
        bot_message_content = ""
        image_data_base64 = None
        image_mime_type_from_tool = None
        video_data_base64 = None
        video_mime_type_from_tool = None
        
        try:
            # First invocation: LLM decides whether to call a tool or respond directly
            logger.info(f"➡️ SENDING TO LLM ({model_name_for_log}) for initial decision - {user_email}")
            response_from_llm = await current_llm.ainvoke(langchain_messages)
            
            # Special handling for image generation requests
            if original_question and any(keyword in original_question.lower() for keyword in ['draw', 'generate image', 'create picture', 'show me']):
                # Check if the response includes code-like patterns referring to image generation but didn't use proper tool calls
                content_str = str(response_from_llm.content) if hasattr(response_from_llm, 'content') else ""
                
                if ('generate_image_tool' in content_str and 'tool_calls' not in dir(response_from_llm)) or not getattr(response_from_llm, 'tool_calls', None):
                    logger.info(f"🛠️ MANUAL TOOL EXTRACTION - LLM mentioned tool but didn't call it properly")
                    # Extract the prompt from the response text
                    import re
                    prompt_match = re.search(r"prompt=['\"]([^'\"]+)['\"]", content_str)
                    
                    if prompt_match:
                        extracted_prompt = prompt_match.group(1)
                        logger.info(f"📝 EXTRACTED PROMPT: '{extracted_prompt}'")
                        
                        # Create a simplified version of the tool call
                        simplified_prompt = self._simplify_image_prompt(extracted_prompt, original_question)
                        logger.info(f"✨ SIMPLIFIED PROMPT: '{simplified_prompt}'")
                        
                        # Generate image using the extracted prompt
                        tool_result = self._generate_image_from_hf_api(simplified_prompt)
                        
                        if tool_result["status"] == "success":
                            image_data_base64 = tool_result["image_data_base64"]
                            image_mime_type_from_tool = tool_result.get("image_mime_type", "image/jpeg")
                            bot_message_content = f"Here's an image of {original_question.lower().replace('draw', '').replace('create', '').replace('generate', '').replace('show me', '').strip()}"
                            
                            # Skip further processing
                            return {
                                "bot_message_content": bot_message_content,
                                "image_data_base64": image_data_base64,
                                "image_mime_type": image_mime_type_from_tool,
                                "video_data_base64": None,
                                "video_mime_type": None,
                                "websearch_info": websearch_info
                            }
            
            if response_from_llm.tool_calls:
                # Log the LLM's decision to call a tool and the prompt it generated
                logger.info(f"⚙️ LLM ({model_name_for_log}) DECIDED TO CALL TOOL - {user_email}")
                
                langchain_messages.append(response_from_llm) # Add LLM's tool call to history
                
                for tool_call in response_from_llm.tool_calls:
                    tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                    tool_args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                    tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                    
                    if tool_name == "generate_image_tool":
                        # Get the original LLM-generated prompt
                        original_prompt = tool_args['prompt']
                        
                        # Simplify the prompt to be more faithful to the user's request
                        simplified_prompt = self._simplify_image_prompt(original_prompt, original_question)
                        
                        # Log both the original and simplified prompts
                        logger.info(f"🖼️ ORIGINAL PROMPT FROM LLM: '{original_prompt}'")
                        logger.info(f"✨ SIMPLIFIED PROMPT FOR IMAGE: '{simplified_prompt}'")
                        
                        # Use the simplified prompt for image generation
                        tool_result = self._generate_image_from_hf_api(simplified_prompt)
                        
                        # Add the original prompt to the result for reference
                        if isinstance(tool_result, dict):
                            tool_result["original_prompt"] = original_prompt
                        
                        langchain_messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_id
                        ))
                        
                        if tool_result["status"] == "success":
                            image_data_base64 = tool_result["image_data_base64"]
                            image_mime_type_from_tool = tool_result.get("image_mime_type", "image/jpeg")
                        
                    elif tool_name == "generate_video_tool":
                        tool_result = self._generate_video_from_hf_api(tool_args['prompt'])
                        
                        langchain_messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_id
                        ))
                        
                        if tool_result["status"] == "success":
                            video_data_base64 = tool_result["video_data_base64"]
                            video_mime_type_from_tool = tool_result.get("video_mime_type", "video/mp4")
                
                # Second invocation: LLM formulates final response based on tool output
                logger.info(f"➡️ SENDING TO LLM ({model_name_for_log}) for final response after tool execution - {user_email}")
                final_response_from_llm = await current_llm.ainvoke(langchain_messages)
                
                # Handle content whether it's string or list
                content = final_response_from_llm.content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif hasattr(item, 'text'):
                            text_parts.append(item.text)
                        elif isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                    bot_message_content = ' '.join(text_parts).strip()
                else:
                    bot_message_content = str(content).strip()
                    
                logger.info(f"✅ LLM ({model_name_for_log}) FINAL RESPONSE (after tool): '{bot_message_content[:200]}...'")
            
            else:
                # No tool call, it's a regular text response from the LLM
                content = response_from_llm.content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif hasattr(item, 'text'):
                            text_parts.append(item.text)
                        elif isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                    bot_message_content = ' '.join(text_parts).strip()
                else:
                    bot_message_content = str(content).strip()
                    
                logger.info(f"✅ LLM ({model_name_for_log}) DIRECT RESPONSE: '{bot_message_content[:200]}...'")
        
        except Exception as e:
            bot_message_content = f"❌ An error occurred: {str(e)}"
            logger.error(f"💥 LLM/Tool error - {user_email} | {e}")
            image_data_base64 = None
            image_mime_type_from_tool = None
            video_data_base64 = None
            video_mime_type_from_tool = None
        
        return {
            "bot_message_content": bot_message_content,
            "image_data_base64": image_data_base64,
            "image_mime_type": image_mime_type_from_tool,
            "video_data_base64": video_data_base64,
            "video_mime_type": video_mime_type_from_tool,
            "websearch_info": websearch_info
        }
    
    def get_all_user_chats(self, user_thread_ids: List[int], db: Session, user_email: str) -> List[Dict[str, Any]]:
        """Get all chats for user threads"""
        
        chats = db.query(Chat).filter(Chat.thread_id.in_(user_thread_ids)).order_by(Chat.timestamp.asc()).all()
        
        return [
            {
                "id": chat.id,
                "sender": chat.sender,
                "message": chat.message,
                "timestamp": chat.timestamp,
                "image_data_base64": getattr(chat, 'image_data_base64', None),
                "image_mime_type": getattr(chat, 'image_mime_type', 'image/jpeg'),
                "video_data_base64": getattr(chat, 'video_data_base64', None),
                "video_mime_type": getattr(chat, 'video_mime_type', 'video/mp4'),
                "filename": getattr(chat, 'filename', None),
                "file_info": {
                    "filename": getattr(chat, 'filename', None),
                    "file_type": getattr(chat, 'file_type', None),
                    "is_downloadable": getattr(chat, 'filename', None) is not None,
                    "mime_type": getattr(chat, 'image_mime_type', None) or getattr(chat, 'video_mime_type', None) or 'application/octet-stream'
                } if getattr(chat, 'filename', None) else None
            }
            for chat in chats
        ]
    
    def get_thread_chats(self, thread_id: int, db: Session, user_email: str) -> List[Dict[str, Any]]:
        """Get all chats for a specific thread"""
        
        chats = db.query(Chat).filter(Chat.thread_id == thread_id).order_by(Chat.timestamp.asc()).all()
        
        logger.info(f"📊 THREAD CHATS RETRIEVED - {user_email} | Thread ID: {thread_id} ")
        
        chat_data = []
        for chat_entry in chats:
            # Enhance file metadata to support downloading
            file_info = None
            if hasattr(chat_entry, 'filename') and chat_entry.filename:
                file_info = {
                    "filename": chat_entry.filename,
                    "file_type": getattr(chat_entry, 'file_type', None),
                    "is_downloadable": True
                }
                
                # Add mime type from image or video if available
                if getattr(chat_entry, 'image_mime_type', None):
                    file_info["mime_type"] = chat_entry.image_mime_type
                elif getattr(chat_entry, 'video_mime_type', None):
                    file_info["mime_type"] = chat_entry.video_mime_type
                else:
                    # Guess mime type from filename extension
                    extension = chat_entry.filename.lower().split('.')[-1] if '.' in chat_entry.filename else ''
                    mime_map = {
                        'pdf': 'application/pdf',
                        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'doc': 'application/msword',
                        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                        'ppt': 'application/vnd.ms-powerpoint',
                        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        'xls': 'application/vnd.ms-excel',
                        'csv': 'text/csv',
                        'txt': 'text/plain',
                        'json': 'application/json',
                        'md': 'text/markdown',
                        'jpg': 'image/jpeg',
                        'jpeg': 'image/jpeg',
                        'png': 'image/png',
                        'gif': 'image/gif',
                        'mp4': 'video/mp4',
                        'webm': 'video/webm'
                    }
                    file_info["mime_type"] = mime_map.get(extension, 'application/octet-stream')

            chat_dict = {
                "id": chat_entry.id,
                "sender": chat_entry.sender,
                "message": chat_entry.message,
                "timestamp": chat_entry.timestamp,
                "thread_id": chat_entry.thread_id,
                "image_data_base64": getattr(chat_entry, 'image_data_base64', None),
                "image_mime_type": getattr(chat_entry, 'image_mime_type', 'image/jpeg'),
                "video_data_base64": getattr(chat_entry, 'video_data_base64', None),
                "video_mime_type": getattr(chat_entry, 'video_mime_type', 'video/mp4'),
                "filename": getattr(chat_entry, 'filename', None),
                "file_type": getattr(chat_entry, 'file_type', None),  # Add direct file_type access
                "mime_type": file_info.get("mime_type") if file_info else None,  # Add direct mime_type access
                "file_info": file_info  # Add the enhanced file metadata
            }
            chat_data.append(chat_dict)
        
        return chat_data

# Create a singleton instance
chat_service = ChatService()