import os
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import json
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class SQLService:
    def __init__(self):
        """Initialize SQL service for dvdrental database"""
        # Setup dvdrental database connection using environment variable
        self.dvdrental_url = os.getenv("POSTGRES_QUERY_TESTING")
        if not self.dvdrental_url:
            raise ValueError("POSTGRES_QUERY_TESTING not found in .env file")
        
        self.dvdrental_engine = create_engine(self.dvdrental_url, echo=False)
        self.DVDRentalSession = sessionmaker(bind=self.dvdrental_engine, autocommit=False, autoflush=False)
        
        # Cache for database schema
        self._schema_cache = None
        
    
    def get_database_schema(self) -> Dict[str, Any]:
        """Get the complete database schema for dvdrental"""
        if self._schema_cache:
            return self._schema_cache
        
        try:
            inspector = inspect(self.dvdrental_engine)
            schema = {
                "tables": {},
                "relationships": []
            }
            
            # Get all table names
            table_names = inspector.get_table_names()
            
            for table_name in table_names:
                columns = inspector.get_columns(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                primary_keys = inspector.get_pk_constraint(table_name)
                
                schema["tables"][table_name] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col["nullable"],
                            "default": col.get("default")
                        }
                        for col in columns
                    ],
                    "primary_keys": primary_keys["constrained_columns"],
                    "foreign_keys": [
                        {
                            "constrained_columns": fk["constrained_columns"],
                            "referred_table": fk["referred_table"],
                            "referred_columns": fk["referred_columns"]
                        }
                        for fk in foreign_keys
                    ]
                }
            
            self._schema_cache = schema
            logger.info(f"📊 Database schema cached for {len(table_names)} tables")
            return schema
            
        except Exception as e:
            logger.error(f"❌ Failed to get database schema: {e}")
            return {"tables": {}, "relationships": []}
    
    def generate_sql_from_natural_language(self, question: str) -> Dict[str, Any]:
        """Convert natural language question to SQL query using Google Gemini API"""
        try:
            schema = self.get_database_schema()
            
            # Create a concise schema description for the AI
            schema_description = self._create_schema_description(schema)
            
            prompt = f"""
You are a SQL expert working with a PostgreSQL dvdrental database. Convert the following natural language question into a SQL query.

DATABASE SCHEMA:
{schema_description}

RULES:
1. Generate ONLY the SQL query, no explanations
2. Use proper PostgreSQL syntax
3. Use appropriate JOINs when needed
4. Include LIMIT clause if question asks for "top" or "first" results
5. Use proper date/time functions for temporal queries
6. Make sure column names and table names are correct
7. Return only SELECT statements (no INSERT, UPDATE, DELETE)

QUESTION: {question}

SQL QUERY:"""

            # Use Google Gemini API directly
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={google_api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            sql_query = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # Clean up the SQL query
            sql_query = self._clean_sql_query(sql_query)
            
            logger.info(f"🔍 Generated SQL for question: '{question[:50]}...'")
            logger.info(f"📝 SQL Query: {sql_query}")
            
            return {
                "status": "success",
                "sql_query": sql_query,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate SQL from natural language: {e}")
            return {
                "status": "error",
                "message": f"Failed to generate SQL: {str(e)}"
            }
    
    def execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            with self.dvdrental_engine.connect() as connection:
                result = connection.execute(text(sql_query))
                
                # Get column names
                columns = list(result.keys())
                
                # Fetch all rows
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                data = []
                for row in rows:
                    row_dict = {}
                    for i, value in enumerate(row):
                        # Handle different data types
                        if hasattr(value, 'isoformat'):  # DateTime objects
                            row_dict[columns[i]] = value.isoformat()
                        elif isinstance(value, (int, float, str, bool)) or value is None:
                            row_dict[columns[i]] = value
                        else:
                            row_dict[columns[i]] = str(value)
                    data.append(row_dict)
                
                logger.info(f"✅ SQL query executed successfully, returned {len(data)} rows")
                
                return {
                    "status": "success",
                    "data": data,
                    "columns": columns,
                    "row_count": len(data),
                    "sql_query": sql_query
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to execute SQL query: {e}")
            return {
                "status": "error",
                "message": f"SQL execution error: {str(e)}",
                "sql_query": sql_query
            }
    
    def answer_natural_language_question(self, question: str) -> Dict[str, Any]:
        """Complete pipeline: natural language question -> SQL -> results -> natural language answer"""
        try:
            # Step 1: Generate SQL from natural language
            sql_result = self.generate_sql_from_natural_language(question)
            
            if sql_result["status"] != "success":
                return sql_result
            
            sql_query = sql_result["sql_query"]
            
            # Step 2: Execute SQL query
            query_result = self.execute_sql_query(sql_query)
            
            if query_result["status"] != "success":
                return query_result
            
            # Step 3: Generate natural language answer
            answer_result = self._generate_natural_language_answer(
                question, sql_query, query_result["data"]
            )
            
            return {
                "status": "success",
                "question": question,
                "sql_query": sql_query,
                "data": query_result["data"],
                "row_count": query_result["row_count"],
                "natural_language_answer": answer_result["answer"],
                "execution_time": "< 1s"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to answer natural language question: {e}")
            return {
                "status": "error",
                "message": f"Failed to process question: {str(e)}"
            }
    
    def _create_schema_description(self, schema: Dict[str, Any]) -> str:
        """Create a concise schema description for AI prompt"""
        description = "TABLES:\n"
        
        for table_name, table_info in schema["tables"].items():
            description += f"\n{table_name}:\n"
            for col in table_info["columns"]:
                description += f"  - {col['name']} ({col['type']})\n"
            
            if table_info["foreign_keys"]:
                description += "  Foreign Keys:\n"
                for fk in table_info["foreign_keys"]:
                    description += f"    - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}\n"
        
        return description
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and validate SQL query"""
        # Remove markdown code blocks if present
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Remove any trailing semicolons
        sql_query = sql_query.rstrip(';')
        
        # Ensure it's a SELECT statement
        if not sql_query.upper().strip().startswith('SELECT'):
            raise ValueError("Only SELECT statements are allowed")
        
        return sql_query
    
    def _generate_natural_language_answer(self, question: str, sql_query: str, data: List[Dict]) -> Dict[str, Any]:
        """Generate natural language answer from SQL results using Google Gemini API"""
        try:
            # Limit data for prompt if too large
            sample_data = data[:10] if len(data) > 10 else data
            
            prompt = f"""
Based on the SQL query results, provide a clear and concise natural language answer to the user's question.

ORIGINAL QUESTION: {question}
SQL QUERY: {sql_query}
RESULT COUNT: {len(data)} rows
SAMPLE DATA: {json.dumps(sample_data, indent=2, default=str)}

Provide a natural, conversational answer that:
1. Directly answers the user's question
2. Includes relevant data from the results
3. Is easy to understand
4. Mentions the total count if relevant

ANSWER:"""

            # Use Google Gemini API directly
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={google_api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            answer = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            return {
                "status": "success",
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate natural language answer: {e}")
            return {
                "status": "error",
                "answer": f"I found {len(data)} results for your question, but couldn't generate a natural language summary."
            }
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions users can ask"""
        return [
            "How many customers do we have?",
            "What are the top 5 most rented movies?",
            "Which customer has spent the most money?",
            "How many movies are in each category?",
            "What is the average rental duration?",
            "Which actors have appeared in the most films?",
            "What are the most popular film categories?",
            "How much revenue was generated last month?",
            "Which stores have the highest rental volume?",
            "What movies are currently overdue?"
        ]
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connection"""
        try:
            with self.dvdrental_engine.connect() as connection:
                result = connection.execute(text("SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public'"))
                table_count = result.fetchone()[0]
                
                return {
                    "status": "success",
                    "message": f"Successfully connected to dvdrental database with {table_count} tables"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to connect to dvdrental database: {str(e)}"
            }
