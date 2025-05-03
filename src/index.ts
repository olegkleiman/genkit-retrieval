// from https://firebase.google.com/docs/genkit/rag

import { googleAI, gemini20Flash, gemini25FlashPreview0417 } from '@genkit-ai/googleai';
import { textEmbedding004, vertexAI } from '@genkit-ai/vertexai';
import { genkit, z } from 'genkit/beta';
import { 
    devLocalIndexerRef, 
    devLocalRetrieverRef,
    devLocalVectorstore
 } from '@genkit-ai/dev-local-vectorstore';
 import {
    // llama31,
    llama32,
    vertexAIModelGarden,
  } from '@genkit-ai/vertexai/modelgarden';
// import { cohereReranker, cohere } from '@genkit-ai/cohere';
import { Document, CommonRetrieverOptionsSchema } from 'genkit/retriever';
// import { RankedDocument, CommonRerankerOptionsSchema } from 'genkit/reranker';
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
    try {
        const pdfFile = path.resolve(filePath);
        const dataBuffer = await readFile(pdfFile);
        const data = await pdf(dataBuffer);
        return data.text;
    } catch (error) {
        logger.error('Error extracting text from PDF:', error);
        throw error;
    }
}

logger.setLogLevel('debug');

googleAI({ apiKey: process.env.GOOGLE_API_KEY });

const ai = genkit({

    plugins: [
        googleAI(),
        // vertexAIModelGarden({
        //     location: 'us-central1',
        //     models: [llama32]
        // }),
        vertexAI(), 
        devLocalVectorstore([
            {
                indexName: 'menuQA',
                embedder: textEmbedding004,            
            }
        ])
    ],
    promptDir: './llm_prompts',
    model: gemini25FlashPreview0417// gemini20Flash,
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

const igReranker = ai.defineReranker(
    {
      name: 'custom/reranker',
      configSchema: z.object({
        k: z.number().optional(),
      }),
    },
    async (query, documents, options) => {

        const queryText = query.text?.toLowerCase() || '';
        logger.info(`Retriever received query: ${queryText}`);

        // 1. Extract keywords from the query (simple split by space).
        const queryKeywords = new Set(queryText.split(/\s+/).filter(Boolean)); // Get unique keywords

        const rerankedDocs = documents.map((doc) => {
            // 2. For each document, count how many query keywords appear in its content.
            const docText = doc.text?.toLowerCase() || '';
            let score = 0;
            queryKeywords.forEach(keyword => {
                if (docText.includes(keyword)) {
                    // 3. Assign the count as the score.
                    score++;
                }
            });

            return {
                ...doc,
                metadata: { ...doc.metadata, score },
            };
        });

        const sortedDocs = rerankedDocs
                                .sort((a, b) => b.metadata.score - a.metadata.score)
                                .slice(0, options.k || 3);
        return {
            documents: sortedDocs,
        };
    }
  );
 
const advancedMenuRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
    // 'k' is already in CommonRetrieverOptionsSchema, but you could add others:
    preRerankK: z.number().max(1000).optional().describe("Number of documents to retrieve before potential reranking"),
    customFilter: z.string().optional().describe("A custom filter string"),
});

const advancedRetriever = ai.defineRetriever({
        name: 'custom/advancedRetriever',
        info: { label: 'Configurable Retriever' },
        configSchema: advancedMenuRetrieverOptionsSchema,
    }, async( query: Document, options: z.infer<typeof advancedMenuRetrieverOptionsSchema> ) => {
        
        logger.info(`Retriever received query: ${query.text}`);

        const initialK = options.preRerankK || 10; // Default to 10 if not provided
        const finalK = options.k ?? 3; // Default final number of docs to 3 if k is not set
        
        const docs = await ai.retrieve({ // kNN or ANN inside
            retriever: menuRetriever,
            query: query,
            options: {
                k: initialK
            }
        });

        const rerankedDocs = await ai.rerank({
            reranker: igReranker,
            query: query,
            documents: docs,
            options:  { k: finalK }
        });

        return {
            documents: rerankedDocs
        };
    });

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

export const promptFlow = ai.defineFlow(
    {
        name: "promptFlow",
        inputSchema: z.string().describe("Prompt input"),
    },
    async (input: string) => {

        // retrieve relevant documents. Uses kNN internally, then re-ranks the retrieved docs
        const docs = await ai.retrieve({
            retriever: advancedRetriever, //use the custom retriever
            query: input,
            options: {
                k: 3,
                preRerankK: 10,
                customFilter: "words count > 5",
            }
        });

        // This is Dotprompt (see https://github.com/google/dotprompt)
        const prompt = ai.prompt('tools_agent'); // .prompt extension will be added automatically
        const promptResponse = await prompt({ // Full conversation turn with LLM here. After it completes, 'text' contains the response text
            // Prompt input
            input,
            docs
        }); 
        console.log("Response to prompt: ", promptResponse.text);
        return promptResponse;
    }
)

export const RAGFlow = ai.defineFlow(
    {
        name: "RAGFlow",
        inputSchema: z.string(),        
    },
    async (input: string) => {

        // const prompt = ai.prompt('tools_agent');
        // const promptResponse = await prompt({ // Full conversation turn with LLM here. After it completes, 'text' contains the response text
        //     // Prompt input
        //     input,
        // }); 
        // console.log("Response to prompt: ", promptResponse.text);

        // retrieve relevant documents. Uses kNN internally, then re-ranks the retrieved docs
        const docs = await ai.retrieve({
            retriever: advancedRetriever, //use the custom retriever
            query: input,
            options: {
                k: 3,
                preRerankK: 10,
                customFilter: "words count > 5",
            }
        });

        // generate a response
        const llmResponse =  ai.generate({
            tools: [getEvents],
            // returnToolRequests: true,
            // prompt: `
            //     You are acting as a helpful AI assistant that can answer 
            //     questions about the topic covered in the article attached and in question's body.

            //     Question: ${input}`,
            prompt: `
                Question: ${input}`,                
            docs
        });

        const toolRequests = (await llmResponse).toolRequests;
        console.log("Tool requests: ", toolRequests);

        return llmResponse;
    }
);
 
const getEvents = ai.defineTool(
    {
        name: "eventsTool",
        description: 'Gets the current events in a given location',
        inputSchema: z.object({ 
          location: z.string().describe('The location to get the current events for')
        }),
        outputSchema: z.string()
    },
    async (input, {context, interrupt, resumed}) => {
        console.log('Input:', input);
        return "List of events in " + input.location + ":\n Event 1\n, Event 2\n, Event 3.";
    }
);

startFlowServer({
    flows: [RAGFlow, indexFlow, promptFlow],
    port: 3400,
});