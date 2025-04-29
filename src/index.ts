// from https://firebase.google.com/docs/genkit/rag

import { googleAI, gemini20Flash } from '@genkit-ai/googleai';
import { textEmbedding004, vertexAI } from '@genkit-ai/vertexai';
import { genkit, z } from 'genkit/beta';
import { 
    devLocalIndexerRef, 
    devLocalRetrieverRef,
    devLocalVectorstore
 } from '@genkit-ai/dev-local-vectorstore';
import { Document } from 'genkit/retriever';
import { logger } from 'genkit/logging';
import { chunk } from 'llm-chunk';
import { startFlowServer } from '@genkit-ai/express';
import path from 'path';
import { readFile } from 'fs/promises';
import pdf from 'pdf-parse';
import dotenv from 'dotenv';
dotenv.config();

console.log("Hello from Genkit!");

async function extractTextFromPdf(filePath: string) {
    const pdfFile = path.resolve(filePath);
    const dataBuffer = await readFile(pdfFile);
    const data = await pdf(dataBuffer);
    return data.text;
}

logger.setLogLevel('debug');

googleAI({ apiKey: process.env.GOOGLE_API_KEY });

const ai = genkit({

    plugins: [
        googleAI(), 
        vertexAI(), 
        devLocalVectorstore([
            {
                indexName: 'menuQA',
                embedder: textEmbedding004,            
            }
        ])
    ],
    model: gemini20Flash,
});


const chunkingConfig = {
    minLength: 1000,
    maxLength: 2000,
    splitter: 'sentence',
    overlap: 100,
    delimiters: '',
  } as any;

const menuPdfIndexer = devLocalIndexerRef('menuQA')  
const menuRetriever = devLocalRetrieverRef('menuQA')

const indexFlow = ai.defineFlow({
        name: "indexFlow",
        inputSchema: z.string().describe("PDF file path"),
    },
    async(filePath: string) => {
        filePath = path.resolve(__dirname, `../docs/${filePath}`);

        // Read the pdf.
        const pdfTxt = await ai.run('extract-text', () =>
            extractTextFromPdf(filePath)
        );

        // Divide the pdf text into segments.
        const chunks = await ai.run('chunk-it', async () =>
            chunk(pdfTxt, chunkingConfig)
        );

        // Convert chunks of text into documents to store in the index.
        const documents = chunks.map((text) => {
                return Document.fromText(text, { filePath });
            });

        // Add documents to the index.
        await ai.index({
            indexer: menuPdfIndexer,
            documents
        });
    }
);

export const menuQAFlow = ai.defineFlow(
    {
        name: "menuQAFlow",
        inputSchema: z.string(),
    },
    async (input: string) => {
        
        // retrieve relevant documents
        const docs = await ai.retrieve({
            retriever: menuRetriever,
            query: input,
            options: { k : 3 }
        });

        // generate a response
        const text =  ai.generate({
            model: gemini20Flash,
            prompt: `
                You are acting as a helpful AI assistant that can answer 
                questions about the food available on the menu at Genkit Grub Pub.

                Use only the context provided to answer the question.
                If you don't know, do not make up an answer.
                Do not add or change items on the menu.

                Question: ${input}`,
            docs
        });

        return text;
    }
);

startFlowServer({
    flows: [menuQAFlow, indexFlow],
    port: 3400,
});