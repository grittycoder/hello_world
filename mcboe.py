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
