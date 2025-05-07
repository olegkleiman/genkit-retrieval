import { genkit, z } from 'genkit/beta';
import { Document, CommonRetrieverOptionsSchema } from 'genkit/retriever';
import { logger } from 'genkit/logging';
import { readFile } from 'fs/promises';
import { Parser as XmlParser } from 'xml2js';
import path from 'path';

// Helper function to recursively extract text from the parsed XML object
export function extractTextFromParsedXml(node: any): string {
  let text = '';
  if (typeof node === 'string') {
    // Append string content, trimmed, with a space
    return node.trim() + ' ';
  }
  if (Array.isArray(node)) {
    // If it's an array, process each item
    for (const item of node) {
      text += extractTextFromParsedXml(item);
    }
  } else if (typeof node === 'object' && node !== null) {
    // If it's an object, process each property
    for (const key in node) {
      // Skip attributes (usually under '$') or other special keys if needed
      if (key === '$') continue;
      text += extractTextFromParsedXml(node[key]);
    }
  }
  return text;
}