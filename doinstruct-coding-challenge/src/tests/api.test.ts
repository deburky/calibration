import { createServer } from 'http';
import { Server } from 'http';
import { Card, ApiResponse } from '../types/types';
import { generateCardsMock } from '../services/llmService';
import { generatePDF } from '../services/pdfService';

const TEST_TIMEOUT = 10000;
let server: Server;

beforeAll(() => {
  // Set test environment
  process.env.NODE_ENV = 'test';
  
  server = createServer((req, res) => {
    let body = '';
    
    req.on('data', chunk => {
      body += chunk.toString();
    });
    
    req.on('end', async () => {
      console.log('Received request body:', body);
      
      try {
        const data = JSON.parse(body);
        
        // Validate required fields
        if (!data.moduleTitle || !data.moduleLanguage) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Missing required fields' }));
          return;
        }
        
        // Validate language
        const validLanguages = ['en', 'de', 'es', 'fr'];
        if (!validLanguages.includes(data.moduleLanguage)) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid language' }));
          return;
        }
        
        console.log('Generating cards for request:', data);
        const cards = await generateCardsMock(data.moduleTitle, data.moduleLanguage, data.instructions);
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ cards }));
      } catch (error) {
        console.error('Error details:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Internal server error' }));
      }
    });
  });
  
  server.listen(3001, () => {
    console.log('Server running at http://localhost:3001');
  });
});

afterAll(() => {
  server.close();
});

describe('Lesson Card Generation API', () => {
  describe('API Endpoint Validation', () => {
    it('should validate required fields', async () => {
      const response = await fetch('http://localhost:3001', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      
      expect(response.status).toBe(400);
      const data = await response.json() as ApiResponse;
      expect(data.error).toBe('Missing required fields');
    });
    
    it('should validate module language', async () => {
      const response = await fetch('http://localhost:3001', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          moduleTitle: 'Test',
          moduleLanguage: 'invalid'
        })
      });
      
      expect(response.status).toBe(400);
      const data = await response.json() as ApiResponse;
      expect(data.error).toBe('Invalid language');
    });
    
    it('should accept valid languages (en/de/es/fr)', async () => {
      const languages = ['en', 'de', 'es', 'fr'];
      const testCases = languages.map(lang => ({
        moduleTitle: 'Test',
        moduleLanguage: lang
      }));
      
      const results = await Promise.all(
        testCases.map(async (testCase) => {
          const response = await fetch('http://localhost:3001/generate-cards', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(testCase),
          });
          const data = await response.json();
          return data as ApiResponse;
        })
      );
      
      results.forEach((result) => {
        expect(result.cards).toBeDefined();
        expect(Array.isArray(result.cards)).toBe(true);
        expect(result.cards.length).toBeGreaterThan(0);
      });
    }, TEST_TIMEOUT * 2);
  });
  
  describe('Response Format', () => {
    it('should return cards in correct JSON structure', async () => {
      const response = await fetch('http://localhost:3001', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          moduleTitle: 'Test',
          moduleLanguage: 'en'
        })
      });
      
      expect(response.status).toBe(200);
      const data = await response.json() as ApiResponse;
      
      // Verify response structure
      expect(data.cards).toBeDefined();
      expect(Array.isArray(data.cards)).toBe(true);
      expect(data.cards.length).toBeGreaterThan(0);
      
      data.cards.forEach((card: Card) => {
        expect(card.title).toBeDefined();
        expect(card.content).toBeDefined();
        expect(typeof card.title).toBe('string');
        expect(typeof card.content).toBe('string');
      });
    }, TEST_TIMEOUT);
    
    it('should respect content length limits', async () => {
      const response = await fetch('http://localhost:3001', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          moduleTitle: 'Test',
          moduleLanguage: 'en'
        })
      });
      
      const data = await response.json() as ApiResponse;
      
      data.cards.forEach((card: Card) => {
        if (card.title) {
          expect(card.title.length).toBeLessThanOrEqual(100);
        }
        if (card.content) {
          expect(card.content.length).toBeLessThanOrEqual(500);
        }
      });
    }, TEST_TIMEOUT);
  });
  
  describe('Content Generation', () => {
    it('should generate cards in the specified language', async () => {
      const response = await fetch('http://localhost:3001', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          moduleTitle: 'Test',
          moduleLanguage: 'de'
        })
      });
      
      expect(response.status).toBe(200);
      const data = await response.json() as ApiResponse;
      expect(data.cards).toBeDefined();
      expect(Array.isArray(data.cards)).toBe(true);
      expect(data.cards.length).toBeGreaterThan(0);
      
      // Verify German content
      const germanKeywords = ['Sicherheit', 'Gefahr', 'Vorsicht', 'Warnung', 'Schutz'];
      const hasGermanContent = data.cards.some(card => 
        germanKeywords.some(keyword => 
          card.title.includes(keyword) || card.content.includes(keyword)
        )
      );
      expect(hasGermanContent).toBe(true);
    }, TEST_TIMEOUT);
    
    it('should handle optional document URL', async () => {
      const response = await fetch('http://localhost:3001', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          moduleTitle: 'Test',
          moduleLanguage: 'en',
          documentUrl: 'example.pdf'
        })
      });
      
      expect(response.status).toBe(200);
      const data = await response.json() as ApiResponse;
      expect(data.cards.length).toBeGreaterThan(0);
    }, TEST_TIMEOUT);

    it('should handle PDF content extraction', async () => {
      const response = await fetch('http://localhost:3001/generate-cards', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          moduleTitle: 'Diesel Safety',
          moduleLanguage: 'de',
          documentUrl: 'DieselKraftStoff.pdf'
        }),
      });

      expect(response.status).toBe(200);
      const data = await response.json() as ApiResponse;
      expect(data.cards).toBeDefined();
      expect(Array.isArray(data.cards)).toBe(true);
      expect(data.cards.length).toBeGreaterThan(0);
      
      // Check if the content reflects the PDF content
      const hasRelevantContent = data.cards.some(card => 
        card.content.toLowerCase().includes('diesel') || 
        card.content.toLowerCase().includes('kraftstoff')
      );
      expect(hasRelevantContent).toBe(true);
    });
  });
  
  describe('PDF Generation', () => {
    it('should generate a PDF file', async () => {
      const cards: Card[] = [
        {
          title: 'Test Card',
          content: 'This is a test card content.'
        }
      ];
      
      const pdfBuffer = await generatePDF(cards);
      expect(pdfBuffer).toBeDefined();
      expect(pdfBuffer instanceof Buffer).toBe(true);
    });
    
    it('should validate required fields for PDF generation', async () => {
      await expect(generatePDF([])).rejects.toThrow('No cards provided');
    });
  });
}); 