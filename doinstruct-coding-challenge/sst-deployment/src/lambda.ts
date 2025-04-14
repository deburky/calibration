import { APIGatewayProxyHandler } from "aws-lambda";
import { generateCardsMock } from "./services/llmService.js";
import { Card } from "./types.js";

// Mock diesel safety PDF content
const mockDieselPDF = `Gefahren für Mensch und Umwelt
Dieselkraftstoff (Flüssigkeit und Dämpfe) ist entzündbar.
Dämpfe und Sprühnebel können mit Luft explosionsfähige Gemische bilden.
Bereits 1% Benzin im Diesel kann das Gemisch leicht entzündbar machen.`;

// Language-specific safety terminology
const safetyTerms = {
  en: [
    "hazardous material",
    "protective equipment",
    "safety regulations",
    "accident prevention",
    "hazard warnings"
  ],
  de: [
    "Gefahrstoff",
    "Schutzausrüstung",
    "Sicherheitsvorschriften",
    "Unfallverhütung",
    "Gefahrenhinweise"
  ],
  es: [
    "material peligroso",
    "equipo de protección",
    "normas de seguridad",
    "prevención de accidentes",
    "advertencias de peligro"
  ],
  fr: [
    "matière dangereuse",
    "équipement de protection",
    "règles de sécurité",
    "prévention des accidents",
    "avertissements de danger"
  ]
};

// Generate mock cards based on input parameters
function generateMockCards(moduleTitle: string, moduleLanguage: string, instructions?: string, documentUrl?: string) {
  // Safety theme based on language
  const terms = safetyTerms[moduleLanguage as keyof typeof safetyTerms] || safetyTerms.en;
  const usePdf = !!documentUrl;
  
  // Create 3-5 cards based on input
  return {
    cards: [
      {
        title: `${moduleTitle} - ${terms[0]}`,
        content: usePdf 
          ? `${mockDieselPDF.split('\n')[0]}. ${mockDieselPDF.split('\n')[1]}` 
          : `Safety card content about ${moduleTitle} in ${moduleLanguage} focusing on ${terms[0]} and ${terms[1]}.`,
        mediaUrl: null
      },
      {
        title: `${terms[1]} - ${moduleTitle}`,
        content: usePdf 
          ? `${mockDieselPDF.split('\n')[2]}. ${mockDieselPDF.split('\n')[3]}` 
          : `Additional information about ${moduleTitle} safety procedures in ${moduleLanguage} with emphasis on ${terms[2]}.${instructions ? ` Following instructions: ${instructions}` : ''}`
      },
      {
        title: `${moduleTitle} - ${terms[3]}`,
        content: `Important safety guidelines for ${moduleTitle} including ${terms[3]} and ${terms[4]}. ${usePdf ? 'Information based on provided documentation.' : ''}`,
        mediaUrl: null
      }
    ]
  };
}

export const handler: APIGatewayProxyHandler = async (event) => {
  try {
    const body = JSON.parse(event.body || "{}");
    
    // Validate required fields
    if (!body.moduleTitle) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: "Module title is required" }),
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      };
    }

    if (!body.moduleLanguage || !["en", "de", "es", "fr"].includes(body.moduleLanguage)) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: "Valid module language (en/de/es/fr) is required" }),
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      };
    }

    // Generate cards using the actual service
    const cards = await generateCardsMock(
      body.moduleTitle,
      body.moduleLanguage,
      body.documentUrl
    );

    return {
      statusCode: 200,
      body: JSON.stringify({ cards }),
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    };
  } catch (error) {
    console.error("Error:", error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: "Internal server error" }),
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    };
  }
};