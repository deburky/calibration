# PY_USAGE.md - Python Equivalent Architecture

This document explains the TypeScript codebase as if it were written in Python, to help ML developers understand the architecture and how it would be structured using Python tools and libraries.

## Overview

This project is a serverless API for generating educational lesson cards using LLM technology (specifically GPT-4). It takes input about a module (title, language, instructions) and generates formatted learning cards with safety-related content.

## Python Equivalent Architecture

### Project Structure

If written in Python, the project would be structured as:

```
doinstruct-coding-challenge/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Equivalent to localServer.ts
│   ├── functions/
│   │   └── generate_cards.py   # Equivalent to generateCards.ts (AWS Lambda handler)
│   ├── services/
│   │   └── llm_service.py      # Equivalent to llmService.ts
│   └── types/
│       └── schemas.py          # Equivalent to types/index.ts (using Pydantic)
├── tests/
│   └── test_api.py             # Equivalent to api.test.ts
├── requirements.txt            # Equivalent to package.json
└── README.md
```

### API Server Implementation

The HTTP server would be built with FastAPI instead of Node's HTTP module:

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.llm_service import generate_cards
from app.types.schemas import GenerateCardsRequest, GenerateCardsResponse

app = FastAPI()

@app.post("/generate-cards", response_model=GenerateCardsResponse)
async def create_cards(request: GenerateCardsRequest):
    # Validation happens automatically through Pydantic
    try:
        cards = await generate_cards(request)
        return {"cards": cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
```

### LLM Service

The LLM service would use the OpenAI Python SDK:

```python
# llm_service.py
import openai
from typing import List, Dict, Any, Optional
import os
from app.types.schemas import Card, GenerateCardsRequest

# Configure OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

SYSTEM_PROMPT = """You are a professional learning content creator..."""

LANGUAGE_INSTRUCTIONS = {
    "de": "Verwende spezifische Sicherheitsterminologie...",
    "en": "Use specific safety terminology...",
    "es": "Utiliza terminología específica de seguridad...",
    "fr": "Utilisez une terminologie de sécurité spécifique..."
}

def create_user_prompt(request: GenerateCardsRequest) -> str:
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(request.module_language, "")
    return f"""Create learning cards about {request.module_title} in {request.module_language}.
    {lang_instruction}
    Instructions: {request.instructions or 'Focus on safety procedures and hazards'}
    Each card should focus on a specific safety aspect or procedure.
    Include practical examples and clear safety guidelines.
    Content should be detailed but easy to understand."""

async def generate_cards(request: GenerateCardsRequest) -> List[Card]:
    # Mock PDF content extraction would be here
    pdf_content = ""
    if request.document_url:
        pdf_content = "Gefahren für Mensch und Umwelt..."
    
    user_prompt = create_user_prompt(request)
    
    # Call OpenAI API
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    
    try:
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")
            
        parsed = json.loads(content)
        if "cards" not in parsed or not isinstance(parsed["cards"], list):
            raise ValueError("Response is not an array of cards")
            
        # Validate and clean up each card
        cards = []
        for card in parsed["cards"]:
            processed_card = Card(
                title=card.get("title", "")[:100] if card.get("title") else None,
                content=card.get("content", "")[:500] if card.get("content") else None,
                media_url=card.get("mediaUrl")
            )
            cards.append(processed_card)
            
        # Ensure at least one field is present in each card
        if not all(card.title or card.content or card.media_url for card in cards):
            raise ValueError("Each card must have at least one field")
            
        return cards
    except Exception as e:
        print(f"Error in generate_cards: {e}")
        raise
```

### Data Models

Python would use Pydantic for data validation and type checking:

```python
# schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class GenerateCardsRequest(BaseModel):
    module_title: str
    module_language: Literal["en", "de", "es", "fr"]
    instructions: Optional[str] = None
    document_url: Optional[str] = None

class Card(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    media_url: Optional[str] = None

class GenerateCardsResponse(BaseModel):
    cards: List[Card]
```

### Testing

Tests would use pytest instead of Jest:

```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

COMMON_GERMAN_WORDS = ["der", "die", "das", "und", "ist", "in", "mit"]
SAFETY_RELATED_GERMAN_WORDS = ["Sicherheit", "Gefahr", "Vorsicht", "Warnung"]

def test_validate_required_fields():
    response = client.post("/generate-cards", json={})
    assert response.status_code == 422  # FastAPI validation error
    errors = response.json()["detail"]
    assert any("module_title" in e["loc"] for e in errors)
    assert any("module_language" in e["loc"] for e in errors)

def test_generate_cards_in_german():
    response = client.post("/generate-cards", json={
        "module_title": "Diesel Fuel Safety",
        "module_language": "de",
        "instructions": "Fokus auf Sicherheitsverfahren und Gefahren",
        "document_url": "example.pdf"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "cards" in data
    assert isinstance(data["cards"], list)
    assert len(data["cards"]) > 0
    
    for card in data["cards"]:
        assert "title" in card
        assert "content" in card
        
        # Check German language and safety terminology
        text = (card["title"] + " " + card["content"]).lower()
        has_common_german = any(word.lower() in text for word in COMMON_GERMAN_WORDS)
        has_safety_words = any(word.lower() in text for word in SAFETY_RELATED_GERMAN_WORDS)
        
        assert has_common_german
        assert has_safety_words
```

## Python ML Integration Notes

The proposed architecture offers several integration points:

1. The LLM service could easily integrate with fine-tuned models by:
   - Switching from OpenAI to Hugging Face's transformers or proprietary inference server
   - Adding additional context or pre/post processing for domain-specific requirements
   - Using contextualized embeddings for improved safety terminology

2. Document processing could be enhanced with:
   - PyPDF2 or pdf2image for PDF parsing
   - NLTK or spaCy for extracting key terms from training materials
   - Custom OCR using Tesseract via pytesseract

3. Evaluation metrics could be added using:
   - ROUGE/BLEU scores for content quality
   - Custom safety terminology detectors
   - Language-specific classifiers for validating outputs