import * as fs from 'fs';
import * as path from 'path';
import pdf from 'pdf-parse';

export async function extractPdfContent(documentUrl: string): Promise<string> {
  try {
    // Construct the full path to the PDF file
    const pdfPath = path.join(process.cwd(), 'pdf_inputs', documentUrl);

    // Check if file exists
    if (!fs.existsSync(pdfPath)) {
      throw new Error(`PDF file not found: ${documentUrl}`);
    }

    // Read the PDF file
    const dataBuffer = fs.readFileSync(pdfPath);

    // Parse the PDF content
    const data = await pdf(dataBuffer);

    // Return the text content
    return data.text;
  } catch (error: unknown) {
    console.error('Error extracting PDF content:', error);
    if (error instanceof Error) {
      throw new Error(`Failed to extract PDF content: ${error.message}`);
    }
    throw new Error('Failed to extract PDF content: Unknown error');
  }
} 