import { Card, GenerateCardsRequest } from '../types';
import OpenAI from 'openai';
import { extractPdfContent } from './pdfContentService';

// Mock responses for testing
const MOCK_RESPONSES = {
  en: {
    cards: [
      {
        title: "Diesel Safety Overview",
        content: "Diesel fuel is flammable and requires careful handling. Always follow proper safety procedures when working with diesel."
      },
      {
        title: "Fire Prevention",
        content: "Keep diesel away from ignition sources. Even 1% gasoline in diesel can make the mixture highly flammable."
      }
    ]
  },
  de: {
    cards: [
      {
        title: "Diesel Sicherheitsübersicht",
        content: "Dieselkraftstoff ist entzündbar und erfordert vorsichtige Handhabung. Befolgen Sie immer die richtigen Sicherheitsverfahren."
      },
      {
        title: "Brandschutz",
        content: "Halten Sie Diesel von Zündquellen fern. Bereits 1% Benzin im Diesel kann das Gemisch leicht entzündbar machen."
      }
    ]
  },
  es: {
    cards: [
      {
        title: "Seguridad del Diesel",
        content: "El combustible diesel es inflamable y requiere un manejo cuidadoso. Siga siempre los procedimientos de seguridad."
      }
    ]
  },
  fr: {
    cards: [
      {
        title: "Sécurité du Diesel",
        content: "Le carburant diesel est inflammable et nécessite une manipulation prudente. Suivez toujours les procédures de sécurité."
      }
    ]
  }
};

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const SYSTEM_PROMPT = `You are a professional learning content creator specializing in safety training materials.
Your task is to generate learning cards that are clear, concise, and focused on safety procedures and hazards.
Each card must contain safety-specific terminology appropriate for the target language.
The response must be a valid JSON object with a 'cards' array.
Each card in the array must have a 'title' and 'content' field.
The content must be detailed, technically accurate, and emphasize safety procedures.`;

type LanguageInstructions = {
  [key in 'de' | 'en' | 'es' | 'fr']: string;
};

function createUserPrompt(request: GenerateCardsRequest): string {
  const languageInstructions: LanguageInstructions = {
    de: 'Verwende unbedingt die folgenden Sicherheitsbegriffe in den Karten: "Sicherheit", "Gefahr", "Vorsicht", "Warnung", "Schutz", "Risiko", "Maßnahmen", "Vorschriften", "Notfall", "Verhalten". Jede Karte muss mindestens zwei dieser Begriffe enthalten.',
    en: 'Use specific safety terminology like "hazardous material", "protective equipment", "safety regulations", "accident prevention", "hazard warnings", etc.',
    es: 'Utiliza terminología específica de seguridad como "material peligroso", "equipo de protección", "normas de seguridad", "prevención de accidentes", "advertencias de peligro", etc.',
    fr: 'Utilisez une terminologie de sécurité spécifique comme "matière dangereuse", "équipement de protection", "règles de sécurité", "prévention des accidents", "avertissements de danger", etc.'
  };

  return `Create learning cards about ${request.moduleTitle} in ${request.moduleLanguage}.
${languageInstructions[request.moduleLanguage as keyof typeof languageInstructions] || ''}
Instructions: ${request.instructions || 'Focus on safety procedures and hazards'}
Each card should focus on a specific safety aspect or procedure.
Include practical examples and clear safety guidelines.
Content should be detailed but easy to understand.`;
}

export async function generateCards(request: GenerateCardsRequest): Promise<Card[]> {
  const { moduleTitle, instructions, moduleLanguage, documentUrl } = request;
  
  // Mock PDF content extraction
  const pdfContent = documentUrl 
    ? `Gefahren für Mensch und Umwelt
Dieselkraftstoff (Flüssigkeit und Dämpfe) ist entzündbar.
Dämpfe und Sprühnebel können mit Luft explosionsfähige Gemische bilden.
Bereits 1% Benzin im Diesel kann das Gemisch leicht entzündbar machen.`
    : "";

  const userPrompt = createUserPrompt(request);

  console.log('Sending prompt to OpenAI:', { systemPrompt: SYSTEM_PROMPT, userPrompt });
  
  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: userPrompt }
    ],
    temperature: 0.7
  });

  try {
    const content = response.choices[0].message.content;
    console.log('OpenAI response:', content);
    
    if (!content) {
      throw new Error('Empty response from OpenAI');
    }

    const parsed = JSON.parse(content);
    console.log('Parsed response:', parsed);
    
    if (!Array.isArray(parsed.cards)) {
      throw new Error('Response is not an array of cards');
    }

    // Validate and clean up each card
    const cards = parsed.cards.map((card: Partial<Card>) => ({
      title: card.title?.slice(0, 100) || undefined,
      content: card.content?.slice(0, 500) || undefined,
      mediaUrl: card.mediaUrl || undefined
    }));

    console.log('Processed cards:', cards);

    // Ensure at least one field is present in each card
    if (!cards.every((card: Card) => card.title || card.content || card.mediaUrl)) {
      throw new Error('Each card must have at least one field (title, content, or mediaUrl)');
    }

    return cards;
  } catch (error) {
    console.error('Error in generateCards:', error);
    throw error;
  }
}

export async function generateCardsMock(moduleTitle: string, moduleLanguage: string, documentUrl?: string): Promise<Card[]> {
  // Use mock responses in test environment
  if (process.env.NODE_ENV === 'test') {
    return MOCK_RESPONSES[moduleLanguage as keyof typeof MOCK_RESPONSES]?.cards || [];
  }

  console.log('Generating cards for request:', { moduleTitle, moduleLanguage, documentUrl });

  try {
    // Extract PDF content if documentUrl is provided
    const pdfContent = documentUrl 
      ? await extractPdfContent(documentUrl)
      : "";

    // Use the extracted content or proceed without it
    const prompt = `Create learning cards for the module "${moduleTitle}" in ${moduleLanguage} language.${pdfContent ? `\nUse this content as reference:\n${pdfContent}` : ''}`;

    const systemPrompt = `You are a professional learning content creator specializing in safety training materials.
Your task is to generate learning cards that are clear, concise, and focused on safety procedures and hazards.
Each card must contain safety-specific terminology appropriate for the target language.
The response must be a valid JSON object with a 'cards' array.
Each card in the array must have a 'title' and 'content' field.
The content must be detailed, technically accurate, and emphasize safety procedures.`;

    const userPrompt = prompt;

    console.log('Sending prompt to OpenAI:', { systemPrompt, userPrompt });

    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      response_format: { type: "json_object" }
    });

    const response = JSON.parse(completion.choices[0].message.content || '{}');
    return response.cards || [];
  } catch (error) {
    console.error('Error details:', error);
    throw error;
  }
}

function getLanguageSpecificPrompt(language: string): string {
  switch (language) {
    case 'de':
      return 'Verwende unbedingt die folgenden Sicherheitsbegriffe in den Karten: "Sicherheit", "Gefahr", "Vorsicht", "Warnung", "Schutz", "Risiko", "Maßnahmen", "Vorschriften", "Notfall", "Verhalten". Jede Karte muss mindestens zwei dieser Begriffe enthalten.';
    case 'es':
      return 'Utiliza terminología específica de seguridad como "material peligroso", "equipo de protección", "normas de seguridad", "prevención de accidentes", "advertencias de peligro", etc.';
    case 'fr':
      return 'Utilisez une terminologie de sécurité spécifique comme "matière dangereuse", "équipement de protection", "règles de sécurité", "prévention des accidents", "avertissements de danger", etc.';
    default:
      return 'Use specific safety terminology like "hazardous material", "protective equipment", "safety regulations", "accident prevention", "hazard warnings", etc.';
  }
} 