export type Language = 'en' | 'de' | 'es' | 'fr';

export interface GenerateCardsRequest {
  moduleTitle: string;
  instructions?: string;
  moduleLanguage: Language;
  documentUrl?: string;
}

export interface Card {
  title?: string;
  content?: string;
  mediaUrl?: string;
}

export interface GenerateCardsResponse {
  cards: Card[];
} 