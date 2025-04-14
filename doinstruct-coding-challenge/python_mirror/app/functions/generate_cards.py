"""
AWS Lambda handler for the generate-cards endpoint.
Equivalent to generateCards.ts in the TypeScript version.
"""

import json
from typing import Any, Dict

from app.services.llm_service import LLMService
from app.models.schemas import GenerateCardsRequest, Settings


async def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    AWS Lambda handler for card generation.
    
    Args:
        event: The Lambda event object
        context: The Lambda context
    
    Returns:
        An API Gateway response object
    """
    try:
        # Check if body exists
        if not event.get("body"):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Request body is required"})
            }
        
        # Parse request body
        request_data = json.loads(event["body"])
        
        # Validate module title
        if not request_data.get("moduleTitle"):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "moduleTitle is required"})
            }
        
        # Validate module language
        if not request_data.get("moduleLanguage"):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "moduleLanguage is required"})
            }
        
        # Validate language value
        valid_languages = ["en", "de", "es", "fr"]
        if request_data.get("moduleLanguage") not in valid_languages:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "moduleLanguage must be one of: en, de, es, fr"})
            }
        
        # Create request object
        request = GenerateCardsRequest(
            module_title=request_data.get("moduleTitle"),
            module_language=request_data.get("moduleLanguage"),
            instructions=request_data.get("instructions"),
            document_url=request_data.get("documentUrl")
        )
        
        # Initialize settings and service
        settings = Settings()
        llm_service = LLMService(settings)
        
        # Generate cards
        cards = await llm_service.generate_cards(request)
        
        # Return response
        return {
            "statusCode": 200,
            "body": json.dumps({"cards": [card.dict(exclude_none=True) for card in cards]})
        }
    
    except Exception as error:
        print(f"Error generating cards: {error}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"})
        }