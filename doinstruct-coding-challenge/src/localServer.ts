import http from 'http';
import { generateCards } from './services/llmService';
import { IncomingMessage, ServerResponse } from 'http';

// Function to create server for testing
export function createServer(port: number = 3000): http.Server {
    const server = http.createServer(async (req: http.IncomingMessage, res: http.ServerResponse) => {
        if (req.method === 'POST' && req.url === '/generate-cards') {
            let body = '';
            
            req.on('data', (chunk: Buffer) => {
                body += chunk.toString();
            });

            req.on('end', async () => {
                try {
                    console.log('Received request body:', body);
                    const request = JSON.parse(body);
                    
                    // Validate required fields
                    const requiredFields = ['moduleTitle', 'moduleLanguage'];
                    const missingFields = requiredFields.filter(field => !request[field]);
                    
                    if (missingFields.length > 0) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ 
                            error: `Missing required fields: ${missingFields.join(', ')}` 
                        }));
                        return;
                    }
                    
                    // Validate module language
                    const validLanguages = ['en', 'de', 'es', 'fr'];
                    if (!validLanguages.includes(request.moduleLanguage)) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ 
                            error: `Invalid moduleLanguage: ${request.moduleLanguage}. Must be one of: ${validLanguages.join(', ')}` 
                        }));
                        return;
                    }

                    console.log('Generating cards for request:', request);
                    const cards = await generateCards(request);
                    console.log('Generated cards:', cards);

                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ cards }));
                } catch (error) {
                    console.error('Error details:', error);
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Internal server error' }));
                }
            });
        } else if (req.method === 'POST' && req.url === '/generate-pdf') {
            // Mock PDF endpoint for testing
            let body = '';
            
            req.on('data', (chunk: Buffer) => {
                body += chunk.toString();
            });

            req.on('end', async () => {
                try {
                    const request = JSON.parse(body);
                    
                    // Validate required fields
                    const requiredFields = ['moduleTitle', 'moduleLanguage'];
                    const missingFields = requiredFields.filter(field => !request[field]);
                    
                    if (missingFields.length > 0) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ 
                            error: `Missing required fields: ${missingFields.join(', ')}` 
                        }));
                        return;
                    }
                    
                    // Validate module language
                    const validLanguages = ['en', 'de', 'es', 'fr'];
                    if (!validLanguages.includes(request.moduleLanguage)) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ 
                            error: `Invalid moduleLanguage: ${request.moduleLanguage}. Must be one of: ${validLanguages.join(', ')}` 
                        }));
                        return;
                    }

                    // Mock PDF generation
                    const buffer = Buffer.from('PDF content');
                    
                    res.writeHead(200, { 
                        'Content-Type': 'application/pdf',
                        'Content-Disposition': 'attachment; filename=lesson-cards.pdf'
                    });
                    res.end(buffer);
                } catch (error) {
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Internal server error' }));
                }
            });
        } else {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Not found' }));
        }
    });
    
    server.listen(port);
    console.log(`Server running at http://localhost:${port}`);
    return server;
}

// Only start the server if this file is run directly
if (require.main === module) {
    createServer(3000);
}