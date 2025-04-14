# DoInstruct Card Generator

A Python-based API for generating educational content using OpenAI API. This project demonstrates how to effectively use LLMs for content generation, with a focus on safety training materials.

## Key Features

- **LLM Integration**: Clean integration with OpenAI's GPT-4 for content generation
- **Multi-language Support**: Generate content in English, German, Spanish, and French
- **Safety Focus**: Specialized in generating safety training materials
- **Type Safety**: Using Pydantic for robust data validation
- **FastAPI**: Modern, fast web framework for serving the API

## Technical Stack

- **Python 3.8+**: Modern Python features and type hints
- **FastAPI**: High-performance web framework
- **Pydantic**: Data validation and settings management
- **OpenAI API**: GPT-4 integration for content generation
- **uv**: Modern Python package installer and resolver

## Project Structure

```
python_mirror/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/              # ML and data models
│   │   ├── __init__.py
│   │   ├── schemas.py       # Pydantic models
│   │   └── settings.py      # Application settings
│   └── services/
│       ├── __init__.py
│       └── llm_service.py   # OpenAI integration
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone and setup:
```bash
git clone <repository-url>
cd doinstruct-coding-challenge/python_mirror
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

3. Configure environment:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the server:
```bash
uvicorn app.main:app --reload
```

2. Test the API:
```bash
curl -X POST http://localhost:8000/generate-cards \
  -H "Content-Type: application/json" \
  -d '{
    "moduleTitle": "Diesel Fuel Safety",
    "moduleLanguage": "de",
    "instructions": "Fokus auf Sicherheitsverfahren und Gefahren"
  }'
```

## ML Components

### Content Generation Pipeline

1. **Input Processing**:
   - Validates input parameters using Pydantic models
   - Handles multi-language support
   - Processes optional PDF content

2. **Prompt Engineering**:
   - System prompt for safety-focused content
   - Language-specific instructions
   - Dynamic prompt generation based on input

3. **LLM Integration**:
   - Clean OpenAI API integration
   - Response parsing and validation
   - Error handling for LLM responses

4. **Output Processing**:
   - Content validation
   - Length constraints
   - Format standardization

## Configuration

The application uses Pydantic's settings management for configuration:

```python
from pydantic import BaseModel

class Settings(BaseModel):
    """Application settings."""
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7)
    max_title_length: int = Field(default=100)
    max_content_length: int = Field(default=500)
```

## Development

1. Install development dependencies:
```bash
uv pip install -r requirements.txt
```

2. Run tests:
```bash
pytest
```

3. Run linting:
```bash
black .
isort .
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT