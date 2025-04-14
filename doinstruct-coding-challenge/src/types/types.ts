export interface Card {
  title: string;
  content: string;
  mediaUrl?: string;
}

export interface ApiResponse {
  cards: Card[];
  error?: string;
}

export interface GenerateCardsRequest {
  moduleTitle: string;
  moduleLanguage: 'en' | 'de' | 'es' | 'fr';
  instructions?: string;
  documentUrl?: string;
}

export interface PdfService {
  extractContent: (url: string) => Promise<string>;
}

export interface LlmService {
  generateCards: (request: GenerateCardsRequest) => Promise<Card[]>;
} 