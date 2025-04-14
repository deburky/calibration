"""
Main application module for the learning card generation service.
This module handles HTTP requests and coordinates the card generation process.
"""

from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import GenerateCardsRequest, Settings
from app.services.llm_service import LLMService, CardGenerationError

# Initialize FastAPI app
app = FastAPI(
    title="Learning Card Generator",
    description="API for generating educational content using OpenAI's GPT-4",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize settings and services
settings = Settings()
llm_service = LLMService(settings)


@app.post("/generate-cards")
async def generate_cards(request: GenerateCardsRequest) -> Dict[str, Any]:
    """Generate learning cards based on the provided request.
    
    Args:
        request: The card generation request parameters
        
    Returns:
        A dictionary containing the generated cards
        
    Raises:
        HTTPException: If there's an error generating the cards
    """
    try:
        # Generate cards using the LLM service
        cards = await llm_service.generate_cards(request)
        
        # Convert cards to dictionary format
        return {
            "cards": [
                {
                    "title": card.title,
                    "content": card.content,
                    "mediaUrl": card.media_url
                }
                for card in cards
            ]
        }
    except CardGenerationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.
    
    Returns:
        A dictionary indicating the service status
    """
    return {"status": "healthy"}