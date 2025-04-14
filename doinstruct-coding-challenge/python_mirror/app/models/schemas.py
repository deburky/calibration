"""
Data models and schemas for the application.
These models define the structure of our data and are used for validation and type safety.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings and configuration.
    
    This model handles all configuration parameters, including LLM settings
    and content generation constraints.
    """
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7)
    max_title_length: int = Field(default=100)
    max_content_length: int = Field(default=500)


class GenerateCardsRequest(BaseModel):
    """Request model for card generation.
    
    This model defines the input parameters for generating learning cards.
    It includes the module title, target language, and optional instructions
    or document URL for context.
    """
    module_title: str = Field(..., description="Title of the learning module")
    module_language: Literal["en", "de", "es", "fr"] = Field(
        ..., description="Target language for the generated content"
    )
    instructions: Optional[str] = Field(
        None, description="Specific instructions for content generation"
    )
    document_url: Optional[str] = Field(
        None, description="URL to a document providing additional context"
    )


class Card(BaseModel):
    """Model representing a single learning card.
    
    Each card contains a title, content, and optional media URL.
    The content is validated to ensure it meets length requirements
    and contains at least one field.
    """
    title: Optional[str] = Field(
        None,
        max_length=100,
        description="Title of the learning card (max 100 characters)"
    )
    content: Optional[str] = Field(
        None,
        max_length=500,
        description="Content of the learning card (max 500 characters)"
    )
    media_url: Optional[str] = Field(
        None,
        description="URL to any associated media (images, videos, etc.)"
    )
    
    def validate_has_content(self) -> bool:
        """Ensure the card has at least one content field.
        
        Returns:
            bool: True if the card has at least one field (title, content, or media_url)
        """
        return bool(self.title or self.content or self.media_url)


class GenerateCardsResponse(BaseModel):
    """Response model containing generated cards.
    
    This model wraps the list of generated cards in a response object.
    """
    cards: List[Card] = Field(..., description="List of generated learning cards") 