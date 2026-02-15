from abc import ABC, abstractmethod
from typing import Any, List, Dict
from pydantic import BaseModel

# 1. The Data Schema: Ensures all data moving through the block is consistent
class ContentPayload(BaseModel):
    """The unified data structure for the MCBOE."""
    content: str
    metadata: Dict[str, Any] = {}
    history: List[str] = []

# 2. The Plugin Interface: The blueprint for all future "complementary" functions
class ContentEnhancer(ABC):
    """Base class for all functions that improve or transform the main content."""
    @abstractmethod
    def transform(self, payload: ContentPayload) -> ContentPayload:
        pass

# 3. The Main Blade: The Master Coding Block Engine
class MasterCodingBlock:
    """The central kernel that coordinates all content processing."""
    
    def __init__(self):
        self.pipeline: List[ContentEnhancer] = []

    def add_enhancer(self, enhancer: ContentEnhancer):
        """Plugs a new complementary function into the master block."""
        self.pipeline.append(enhancer)
        return self

    def process(self, raw_text: str) -> ContentPayload:
        """
        The foundational 'Main Blade' function. 
        It ingests raw data and runs it through the unified enhancement pipeline.
        """
        # Initializing the payload
        payload = ContentPayload(content=raw_text)
        
        # Iterative transformation through all plugged-in functions
        for enhancer in self.pipeline:
            payload = enhancer.transform(payload)
            payload.history.append(enhancer.__class__.__name__)
            
        return payload

# --- Example of a 'Complementary Function' (Plugin) ---

class TextCleaner(ContentEnhancer):
    """A simple plugin to demonstrate the pipeline."""
    def transform(self, payload: ContentPayload) -> ContentPayload:
        payload.content = payload.content.strip().capitalize()
        return payload

import re

# 1. The Data Cleaner Enhancer
class DataCleaner(ContentEnhancer):
    """
    Purifies raw string data by removing non-printable characters, 
    fixing whitespace, and stripping HTML-like tags.
    """
    def transform(self, payload: ContentPayload) -> ContentPayload:
        text = payload.content
        
        # Remove HTML tags (if any)
        text = re.sub(r'<[^>]*>', '', text)
        
        # Normalize whitespace (replace tabs/newlines with single spaces)
        text = re.sub(r'\s+', 's', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        payload.content = text
        return payload

# 2. The Format Transformer Enhancer
class FormatTransformer(ContentEnhancer):
    """
    Converts cleaned text into a structured format.
    Adds metadata regarding word count and creates a 'preview' snippet.
    """
    def transform(self, payload: ContentPayload) -> ContentPayload:
        content = payload.content
        
        # Add transformation metadata
        words = content.split()
        payload.metadata["word_count"] = len(words)
        payload.metadata["is_long_form"] = len(words) > 50
        
        # Transform the content into a 'Structured' string
        preview = " ".join(words[:10]) + "..." if len(words) > 10 else content
        payload.content = f"PREVIEW: {preview} | FULL TEXT: {content}"
        
        return payload

# --- Demonstration of the Unified Block in Action ---

# Initialize the Master Block
mcboe = MasterCodingBlock()

# Plug in our new complementary functions
mcboe.add_enhancer(DataCleaner())
mcboe.add_enhancer(FormatTransformer())

# Run the "Main Blade"
raw_input = "  Hello! <script>alert('bad')</script>   This is the MCBOE testing...   "
final_result = mcboe.process(raw_input)

print(f"Final Content: {final_result.content}")
print(f"Metadata: {final_result.metadata}")
print(f"Process History: {final_result.history}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize the Web App
app = FastAPI(title="Master Coding Block Of Everything (MCBOE) API")

# 2. Setup the Global Engine
# In a real app, this would be initialized once
engine = MasterCodingBlock()
engine.add_enhancer(DataCleaner())
engine.add_enhancer(FormatTransformer())

# 3. Define the Input Format for the API
class ProcessRequest(BaseModel):
    text: str

# 4. The "Web Door" Endpoint
@app.post("/process")
async def process_content(request: ProcessRequest):
    """
    Ingests raw text via HTTP, runs the MCBOE pipeline, 
    and returns the unified payload.
    """
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text field cannot be empty.")
            
        # Trigger the Main Blade
        result = engine.process(request.text)
        
        return {
            "status": "success",
            "data": result.content,
            "metadata": result.metadata,
            "pipeline_path": result.history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# Update your engine initialization
engine = MasterCodingBlock()
engine.add_enhancer(DataCleaner())
engine.add_enhancer(FormatTransformer())
engine.add_enhancer(DatabaseSaver()) # The memory is now active!        

from typing import List

# 1. Add this endpoint to your FastAPI app section
@app.get("/history", response_model=List[dict])
def get_all_history():
    """
    The 'Recall' function. 
    Queries the SQLite memory to return every block ever processed.
    """
    db = SessionLocal()
    try:
        # Fetching all records from the 'records' table
        records = db.query(ProcessedRecord).order_by(ProcessedRecord.processed_at.desc()).all()
        
        # Formatting for the response
        return [
            {
                "id": r.id,
                "content": r.final_content,
                "metadata": r.metadata_json,
                "timestamp": r.processed_at.isoformat()
            }
            for r in records
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory Retrieval Failed: {str(e)}")
    finally:
        db.close()


# 5. Health Check
@app.get("/")
def read_root():
    return {"message": "MCBOE is Online", "version": "1.0.0"}

from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# 1. Database Setup
DATABASE_URL = "sqlite:///./mcboe_memory.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. The Database Model (The 'Memory' Table)
class ProcessedRecord(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True, index=True)
    final_content = Column(String)
    metadata_json = Column(JSON)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create the table
Base.metadata.create_all(bind=engine)

# 3. The Database Saver Enhancer
class DatabaseSaver(ContentEnhancer):
    """
    The 'Memory' plugin. It saves the state of the payload 
    to the SQLite database at the end of the pipeline.
    """
    def transform(self, payload: ContentPayload) -> ContentPayload:
        db = SessionLocal()
        try:
            new_record = ProcessedRecord(
                final_content=payload.content,
                metadata_json=payload.metadata
            )
            db.add(new_record)
            db.commit()
            payload.metadata["db_record_id"] = new_record.id
        finally:
            db.close()
        return payload
        
        
from sqlalchemy import or_

# Add this endpoint to your mcboe.py file

@app.get("/search")
def search_history(query: str):
    """
    The 'Search' function.
    Scans the SQLite memory for records containing the query string
    within the content or metadata.
    """
    if not query or len(query) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters.")

    db = SessionLocal()
    try:
        # We use a case-insensitive search (ILIKE/contains) 
        # scanning the 'final_content' column
        search_results = db.query(ProcessedRecord).filter(
            ProcessedRecord.final_content.contains(query)
        ).order_by(ProcessedRecord.processed_at.desc()).all()

        if not search_results:
            return {"message": f"No records found matching: '{query}'", "results": []}

        return [
            {
                "id": r.id,
                "content": r.final_content,
                "metadata": r.metadata_json,
                "timestamp": r.processed_at.isoformat()
            }
            for r in search_results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    finally:
        db.close()





import collections

class AISummaryEnhancer(ContentEnhancer):
    """
    The 'Intelligence' plugin. 
    Analyzes the text and adds a 'summary' key to the metadata.
    """
    def transform(self, payload: ContentPayload) -> ContentPayload:
        content = payload.content
        sentences = content.split('.')
        
        # Simple Logic: If text is short, the summary is the text.
        # If long, we take the first two sentences as a 'Pseudo-AI' summary.
        if len(sentences) > 2:
            summary = ". ".join(sentences[:2]).strip() + "..."
        else:
            summary = content
            
        # We store the result in metadata so it doesn't overwrite the full text
        payload.metadata["summary"] = summary
        payload.metadata["analysis_version"] = "v1.0-heuristic"
        
        return payload


# New and Improved Pipeline
engine = MasterCodingBlock()
engine.add_envancer(DataCleaner())
engine.add_enhancer(AISummaryEnhancer())  # <-- AI analysis happens before formatting
engine.add_enhancer(FormatTransformer())
engine.add_enhancer(DatabaseSaver())



from sqlalchemy import func
from datetime import datetime, timedelta

@app.get("/dashboard")
def get_dashboard_stats():
    """
    The 'Observation' function.
    Aggregates data from the SQLite memory to provide a 
    high-level overview of the MCBOE's activity.
    """
    db = SessionLocal()
    try:
        # 1. Total count of all records
        total_records = db.query(ProcessedRecord).count()

        # 2. Today's activity
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        records_today = db.query(ProcessedRecord).filter(
            ProcessedRecord.processed_at >= today_start
        ).count()

        # 3. Calculate Average Word Count (Stored in metadata_json)
        # Note: Since SQLite JSON handling varies, we do a simple Python aggregation here
        all_metadata = db.query(ProcessedRecord.metadata_json).all()
        total_words = sum(m[0].get("word_count", 0) for m in all_metadata if m[0])
        avg_words = total_words / total_records if total_records > 0 else 0

        # 4. Engine Status
        active_enhancers = [e.__class__.__name__ for e in engine.pipeline]

        return {
            "system_status": "ONLINE",
            "metrics": {
                "total_processed_blocks": total_records,
                "processed_today": records_today,
                "average_word_depth": round(avg_words, 2)
            },
            "active_pipeline": active_enhancers,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")
    finally:
        db.close()




