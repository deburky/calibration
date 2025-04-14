import { Card } from '../types.js';
import OpenAI from 'openai';
import { extractPdfContent } from './pdfContentService.js';

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