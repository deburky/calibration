import { APIGatewayProxyHandlerV2 } from 'aws-lambda';
import { generateCards } from '../services/llmService';
import { GenerateCardsRequest, GenerateCardsResponse } from '../types';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
  try {
    if (!event.body) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: 'Request body is required' }),
      };
    }

    const request: GenerateCardsRequest = JSON.parse(event.body);

    // Validate required fields
    if (!request.moduleTitle) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: 'moduleTitle is required' }),
      };
    }

    if (!request.moduleLanguage) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: 'moduleLanguage is required' }),
      };
    }

    // Validate language
    const validLanguages = ['en', 'de', 'es', 'fr'];
    if (!validLanguages.includes(request.moduleLanguage)) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: 'moduleLanguage must be one of: en, de, es, fr' }),
      };
    }

    const cards = await generateCards(request);
    const response: GenerateCardsResponse = { cards };

    return {
      statusCode: 200,
      body: JSON.stringify(response),
    };
  } catch (error) {
    console.error('Error generating cards:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Internal server error' }),
    };
  }
}; 