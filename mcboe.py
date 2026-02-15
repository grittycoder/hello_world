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