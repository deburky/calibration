declare module 'pdf-parse' {
  interface PDFData {
    text: string;
    numpages: number;
    info: Record<string, any>;
    metadata: Record<string, any>;
    version: string;
  }

  function PDFParse(dataBuffer: Buffer): Promise<PDFData>;
  export default PDFParse;
} 