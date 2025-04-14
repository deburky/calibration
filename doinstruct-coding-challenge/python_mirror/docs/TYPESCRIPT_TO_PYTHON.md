# TypeScript to Python Translation Guide

This document outlines how the TypeScript codebase was translated to Python, highlighting key differences and implementation choices. This is useful for explaining your approach during follow-up discussions.

## Architecture Comparison

| TypeScript                 | Python                 | Notes                                   |
|----------------------------|------------------------|-----------------------------------------|
| `src/localServer.ts`       | `app/main.py`          | HTTP server -> FastAPI application      |
| `src/types/index.ts`       | `app/models/schemas.py`| TypeScript interfaces -> Pydantic models|
| `src/services/llmService.ts`| `app/services/llm_service.py`| Similar structure, Python idioms |
| `src/functions/generateCards.ts` | `app/functions/generate_cards.py` | Lambda handler translation |
| Jest tests                 | pytest tests           | Similar test structure                  |

## Key Translation Patterns

### Type System

* **TypeScript interfaces → Pydantic models**
  ```typescript
  // TypeScript
  export interface Card {
    title?: string;
    content?: string;
    mediaUrl?: string;
  }
  ```
  
  ```python
  # Python with Pydantic
  class Card(BaseModel):
      title: Optional[str] = None
      content: Optional[str] = None
      media_url: Optional[str] = None
  ```

### Error Handling

* **TypeScript try/catch → Python try/except**
  ```typescript
  // TypeScript
  try {
    const cards = await generateCards(request);
    return { cards };
  } catch (error) {
    console.error('Error:', error);
    return { error: 'Internal server error' };
  }
  ```
  
  ```python
  # Python
  try:
      cards = await llm_service.generate_cards(request)
      return {"cards": cards}
  except Exception as e:
      print(f"Error: {e}")
      raise HTTPException(status_code=500, detail="Internal server error")
  ```

### HTTP Server

* **Node.js HTTP → FastAPI**
  ```typescript
  // TypeScript with Node.js http
  const server = http.createServer();
  server.on('request', async (req, res) => {
    // Handle request
  });
  ```
  
  ```python
  # Python with FastAPI
  app = FastAPI()
  
  @app.post("/generate-cards")
  async def generate_cards(request: GenerateCardsRequest):
      # Handle request
  ```

### Testing

* **Jest → pytest**
  ```typescript
  // TypeScript with Jest
  describe('API Endpoint', () => {
    it('should validate fields', async () => {
      expect(response.status).toBe(400);
    });
  });
  ```
  
  ```python
  # Python with pytest
  class TestAPIEndpoint:
      def test_validate_fields(self):
          assert response.status_code == 400
  ```

## API Differences

* **Request/Response Format**: Same JSON structure, different field naming convention (camelCase → snake_case)
* **Validation**: TypeScript manual validation → Pydantic automatic validation
* **Error Responses**: Similar HTTP status codes, structured error messages

## Implementation Improvements in Python

1. **Settings Management**: Centralized configuration using Pydantic settings 
2. **Type Safety**: More comprehensive type checking with mypy
3. **Class-Based Design**: More object-oriented approach in the LLM service
4. **Better Exception Handling**: Custom exceptions for different error scenarios
5. **Code Organization**: Cleaner separation of concerns

## Development Tooling

| TypeScript               | Python              |
|--------------------------|---------------------|
| npm/yarn                 | pip/uv              |
| tsc (TypeScript compiler)| mypy                |
| Jest                     | pytest              |
| ESLint                   | black, isort, ruff  |

This translation demonstrates how to port a Node.js/TypeScript serverless application to Python while maintaining the same functionality and architecture, but adapting to Python's idioms and best practices.