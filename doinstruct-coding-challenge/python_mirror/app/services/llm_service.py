"""
LLM service for generating educational content using OpenAI's GPT-4.
This module handles the interaction with the OpenAI API and content generation.
"""

import json
from typing import Dict, List

from openai import OpenAI
from app.models.schemas import Card, GenerateCardsRequest, Settings


class LLMService:
    """Service for interacting with OpenAI's GPT-4 for content generation.
    
    This class handles all interactions with the OpenAI API, including
    prompt engineering, response parsing, and error handling.
    """
    
    def __init__(self, settings: Settings):
        """Initialize the LLM service.
        
        Args:
            settings: Application settings including OpenAI API key
        """
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        
        # System prompt for directing the LLM
        self.system_prompt = """You are a professional learning content creator specializing in safety training materials.
Your task is to generate learning cards that are clear, concise, and focused on safety procedures and hazards.
Each card must contain safety-specific terminology appropriate for the target language.
The response must be a valid JSON object with a 'cards' array.
Each card in the array must have a 'title' and 'content' field.
The content must be detailed, technically accurate, and emphasize safety procedures."""
        
        # Language-specific instructions
        self.language_instructions: Dict[str, str] = {
            "de": "Verwende spezifische Sicherheitsterminologie wie \"Gefahrstoff\", \"Schutzausrüstung\", \"Sicherheitsvorschriften\", \"Unfallverhütung\", \"Gefahrenhinweise\", etc.",
            "en": "Use specific safety terminology like \"hazardous material\", \"protective equipment\", \"safety regulations\", \"accident prevention\", \"hazard warnings\", etc.",
            "es": "Utiliza terminología específica de seguridad como \"material peligroso\", \"equipo de protección\", \"normas de seguridad\", \"prevención de accidentes\", \"advertencias de peligro\", etc.",
            "fr": "Utilisez une terminologie de sécurité spécifique comme \"matière dangereuse\", \"équipement de protection\", \"règles de sécurité\", \"prévention des accidents\", \"avertissements de danger\", etc."
        }
    
    def create_user_prompt(self, request: GenerateCardsRequest) -> str:
        """Create a user prompt for the LLM based on the request parameters.
        
        Args:
            request: The card generation request object
            
        Returns:
            A formatted prompt string
        """
        lang_instruction = self.language_instructions.get(request.module_language, "")
        
        return f"""Create learning cards about {request.module_title} in {request.module_language}.
{lang_instruction}
Instructions: {request.instructions or 'Focus on safety procedures and hazards'}
Each card should focus on a specific safety aspect or procedure.
Include practical examples and clear safety guidelines.
Content should be detailed but easy to understand."""
    
    async def extract_pdf_content(self, document_url: str) -> str:
        """Extract content from a PDF document.
        
        This is a mock implementation. In a real scenario, you would use
        PyPDF2, pdf2image, or similar libraries to extract text from PDFs.
        
        Args:
            document_url: URL to the PDF document
            
        Returns:
            Extracted text content
        """
        # Mock implementation
        return """Gefahren für Mensch und Umwelt
Dieselkraftstoff (Flüssigkeit und Dämpfe) ist entzündbar.
Dämpfe und Sprühnebel können mit Luft explosionsfähige Gemische bilden.
Bereits 1% Benzin im Diesel kann das Gemisch leicht entzündbar machen."""
    
    async def generate_cards(self, request: GenerateCardsRequest) -> List[Card]:
        """Generate learning cards using the OpenAI API.
        
        This method handles the entire content generation pipeline:
        1. PDF content extraction (if URL provided)
        2. Prompt creation
        3. API call to OpenAI
        4. Response parsing and validation
        5. Card creation and validation
        
        Args:
            request: The card generation request parameters
            
        Returns:
            A list of generated cards
            
        Raises:
            CardGenerationError: If there's an error in card generation
        """
        try:
            # Extract PDF content if URL is provided
            if request.document_url:
                # Currently not using the pdf_content, but would integrate in a real implementation
                _ = await self.extract_pdf_content(request.document_url)
            
            # Create the user prompt
            user_prompt = self.create_user_prompt(request)
            
            print(f"Sending prompt to OpenAI: {self.system_prompt}, {user_prompt}")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.settings.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.settings.temperature
            )
            
            # Extract and parse content
            content = response.choices[0].message.content
            print(f"OpenAI response: {content}")
            
            if not content:
                raise CardGenerationError("Empty response from OpenAI")
            
            parsed = json.loads(content)
            print(f"Parsed response: {parsed}")
            
            if not isinstance(parsed.get("cards"), list):
                raise CardGenerationError("Response is not an array of cards")
            
            # Validate and clean up each card
            cards = []
            for card_data in parsed["cards"]:
                card = Card(
                    title=card_data.get("title", "")[:self.settings.max_title_length] if card_data.get("title") else None,
                    content=card_data.get("content", "")[:self.settings.max_content_length] if card_data.get("content") else None,
                    media_url=card_data.get("mediaUrl")
                )
                cards.append(card)
            
            print(f"Processed cards: {cards}")
            
            # Ensure at least one field is present in each card
            if not all(card.validate_has_content() for card in cards):
                raise CardGenerationError("Each card must have at least one field (title, content, or media_url)")
            
            return cards
        
        except json.JSONDecodeError as e:
            raise CardGenerationError(f"Invalid JSON response from OpenAI: {str(e)}")
        except Exception as e:
            raise CardGenerationError(f"Error generating cards: {str(e)}")


class CardGenerationError(Exception):
    """Custom exception for card generation errors."""
    pass