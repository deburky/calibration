"""
Tests for the API endpoints.
Equivalent to api.test.ts in the TypeScript version.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Test constants
COMMON_GERMAN_WORDS = [
    "der", "die", "das", "und", "ist", "in", "mit", "für", "bei", "von"
]

SAFETY_RELATED_GERMAN_WORDS = [
    "Sicherheit", "Gefahr", "Vorsicht", "Warnung", "Schutz", "Risiko",
    "Maßnahmen", "Vorschriften", "Notfall", "Verhalten"
]

TEST_TIMEOUT = 60  # seconds

# Create test client
client = TestClient(app)


class TestLessonCardGenerationAPI:
    """Test suite for the lesson card generation API."""
    
    class TestAPIEndpointValidation:
        """Tests for API endpoint validation."""
        
        def test_validate_required_fields(self):
            """Test that required fields are validated."""
            response = client.post("/generate-cards", json={})
            
            # Pydantic validation causes 422 Unprocessable Entity
            assert response.status_code == 422
            data = response.json()
            
            # Check error detail contains field names
            field_errors = [error["loc"][1] for error in data["detail"]]
            assert "module_title" in field_errors
            assert "module_language" in field_errors
        
        def test_validate_module_language(self):
            """Test that module language is validated."""
            response = client.post("/generate-cards", json={
                "module_title": "Test",
                "module_language": "invalid"
            })
            
            assert response.status_code == 422
            data = response.json()
            
            # Check that the error is about enum validation
            assert any("module_language" in error["loc"] for error in data["detail"])
        
        @pytest.mark.parametrize("language", ["en", "de", "es", "fr"])
        def test_accept_valid_languages(self, language):
            """Test that valid languages are accepted."""
            # Note: This test would be mocked in a real scenario to avoid API calls
            response = client.post("/generate-cards", json={
                "module_title": "Test",
                "module_language": language
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "cards" in data
    
    class TestResponseFormat:
        """Tests for API response format."""
        
        def test_cards_json_structure(self):
            """Test that cards have correct JSON structure."""
            # Note: This test would be mocked in a real scenario
            response = client.post("/generate-cards", json={
                "module_title": "Diesel Fuel Safety",
                "module_language": "de",
                "instructions": "Fokus auf Sicherheitsverfahren und Gefahren"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "cards" in data
            assert isinstance(data["cards"], list)
            
            # Verify each card has at least one of title, content, or media_url
            for card in data["cards"]:
                assert any(key in card and card[key] for key in ["title", "content", "media_url"])
        
        def test_content_length_limits(self):
            """Test that content length limits are respected."""
            # Note: This test would be mocked in a real scenario
            response = client.post("/generate-cards", json={
                "module_title": "Diesel Fuel Safety",
                "module_language": "de",
                "instructions": "Fokus auf Sicherheitsverfahren und Gefahren"
            })
            
            data = response.json()
            
            for card in data["cards"]:
                if "title" in card and card["title"]:
                    assert len(card["title"]) <= 100
                if "content" in card and card["content"]:
                    assert len(card["content"]) <= 500
    
    class TestContentGeneration:
        """Tests for content generation."""
        
        def test_generate_cards_in_german(self):
            """Test that cards are generated in German."""
            # Note: This test would be mocked in a real scenario
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
                
                # Check if titles and content contain German words
                text = (card["title"] + " " + card["content"]).lower()
                has_common_german_words = any(word.lower() in text for word in COMMON_GERMAN_WORDS)
                has_safety_words = any(word.lower() in text for word in SAFETY_RELATED_GERMAN_WORDS)
                
                assert has_common_german_words
                assert has_safety_words
        
        def test_handle_document_url(self):
            """Test that document URL is handled correctly."""
            # Note: This test would be mocked in a real scenario
            response = client.post("/generate-cards", json={
                "module_title": "Diesel Fuel Safety",
                "module_language": "de",
                "instructions": "Fokus auf Sicherheitsverfahren und Gefahren",
                "document_url": "example.pdf"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["cards"]) > 0