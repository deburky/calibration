import PDFDocument from 'pdfkit';
import { Card } from '../types';

export async function generatePDF(cards: Card[]): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    try {
      if (!cards || cards.length === 0) {
        throw new Error('No cards provided');
      }

      // Create a new PDF document
      const doc = new PDFDocument({
        size: 'A4',
        margin: 50
      });

      // Collect PDF chunks in a buffer
      const chunks: Buffer[] = [];
      doc.on('data', chunk => chunks.push(chunk));
      doc.on('end', () => resolve(Buffer.concat(chunks)));

      // Add title page
      doc.fontSize(24)
         .text('Lesson Cards', { align: 'center' })
         .moveDown(2);

      // Add each card
      cards.forEach((card, index) => {
        if (index > 0) {
          doc.addPage();
        }

        // Add card title
        doc.fontSize(18)
           .text(card.title || 'Untitled Card', { align: 'center' })
           .moveDown();

        // Add card content
        doc.fontSize(12)
           .text(card.content || '', { align: 'justify' })
           .moveDown();

        // Add media URL if present
        if (card.mediaUrl) {
          doc.fontSize(10)
             .fillColor('blue')
             .text(card.mediaUrl, { align: 'center', link: card.mediaUrl });
        }
      });

      // Finalize the PDF
      doc.end();
    } catch (error) {
      reject(error);
    }
  });
} 