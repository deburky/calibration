# DoInstruct Coding Challenge - Learning Card Generator

This project implements a serverless API using SST and TypeScript that connects to OpenAI's GPT-4 for generating educational content cards, with a focus on safety training materials.

## Author
Denis Burakov
- GitHub: [deburky](https://github.com/deburky)

## Features

- **API Endpoint** for generating learning cards from user input
- **Multi-language Support**: Generate cards in English, German, Spanish, and French 
- **PDF Integration**: Structured to support PDF content extraction (mock implementation)
- **Input Validation**: Robust validation with appropriate error handling
- **LLM Prompt Engineering**: Effective prompting strategies to generate high-quality cards
- **Type Safety**: Full TypeScript implementation with strict types

## Prerequisites

- Node.js (v18 or later)
- AWS Account (for deployment only)
- OpenAI API Key

## Setup and Local Development

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_openai_api_key
```

3. Start the local server:
```bash
npm start
```

4. The API will be available at `http://localhost:3000`

## API Usage

### Generate Cards

**Endpoint:** `POST /generate-cards`

**Request Body:**
```json
{
  "moduleTitle": "Diesel Fuel Safety",
  "moduleLanguage": "en",
  "instructions": "Focus on safety procedures and hazards",
  "documentUrl": "https://example.com/document.pdf"
}
```

**Response:**
```json
{
  "cards": [
    {
      "title": "Diesel Fuel Hazards",
      "content": "Diesel fuel is flammable and can form explosive mixtures with air. Even 1% gasoline contamination can make it more flammable.",
      "mediaUrl": null
    },
    {
      "title": "Safety Precautions",
      "content": "Always ensure good ventilation when handling diesel fuel. Keep containers closed and store in cool, dry places.",
      "mediaUrl": null
    }
  ]
}
```

## Required Fields
- `moduleTitle`: The title of the learning module (string)
- `moduleLanguage`: The language for the cards (one of: 'en', 'de', 'es', 'fr')

## Optional Fields
- `instructions`: Additional guidance for card generation
- `documentUrl`: URL to a PDF document (currently mocked)

## Testing

Run the test suite:
```bash
npm test
```

For continuous testing during development:
```bash
npm run test:watch
```

## AWS Deployment

This project provides two deployment options:

### Option 1: Deploy the main project

1. Install SST if not already installed:
```bash
npm install -g sst
```

2. Configure AWS credentials:
```bash
aws configure
```

3. Deploy to AWS:
```bash
npx sst deploy
```

4. After deployment, you'll receive an API endpoint URL in the console output.

### Option 2: Deploy the mock version (no OpenAI API required)

The project includes a separate `sst-deployment` directory with a mock implementation that doesn't require an OpenAI API key. This is ideal for testing the deployment process.

1. Navigate to the deployment directory:
```bash
cd sst-deployment
```

2. Install dependencies:
```bash
npm install
```

3. Deploy to AWS:
```bash
npx sst deploy
```

The mock implementation includes:
- Complete API interface matching the requirements
- Mock data generation based on input parameters
- Simulated PDF content based on the diesel safety example
- Support for all required languages

## Architecture

The API is built with:
- **SST**: Infrastructure as code for AWS deployment
- **Express**: Local development server
- **TypeScript**: Type-safe code
- **OpenAI API**: For content generation using GPT-4
- **Jest**: Test framework

## Prompt Engineering Strategy

The implementation uses a two-part prompt structure:

1. **System Prompt**:
```
You are a professional learning content creator specializing in safety training materials.
Your task is to generate learning cards that are clear, concise, and focused on safety procedures and hazards.
Each card must contain safety-specific terminology appropriate for the target language.
The response must be a valid JSON object with a 'cards' array.
Each card in the array must have a 'title' and 'content' field.
```

2. **User Prompt**:
- Incorporates the module title and language
- Includes specific safety terminology
- Adapts based on provided instructions
- Maintains consistent formatting

## Project Structure

```
├── src/
│   ├── functions/            # Lambda functions
│   │   └── generateCards.ts  # Main card generation endpoint
│   ├── services/             # Modular services
│   │   ├── llmService.ts     # OpenAI integration
│   │   └── pdfService.ts     # PDF extraction (mock)
│   ├── types/                # TypeScript type definitions
│   ├── tests/                # Test suite
│   └── localServer.ts        # Express server for local development
├── sst.config.ts             # SST deployment configuration
└── python_mirror/            # Python implementation (supplementary)
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (missing required fields or invalid input)
- 500: Internal Server Error

Error responses include a message explaining the issue:
```json
{
  "error": "Error message here"
}
```