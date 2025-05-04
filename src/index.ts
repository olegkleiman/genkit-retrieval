// from https://firebase.google.com/docs/genkit/rag

import crypto from 'crypto';
import { googleAI } from '@genkit-ai/googleai';
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

import { getEvents } from './tools';
import { ai } from './genkit';

const getHash = (text: string): string => {
    return crypto.createHash('sha256').update(text).digest('hex');
}

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
logger.debug("Hello from Genkit!");

googleAI({ apiKey: process.env.GOOGLE_API_KEY });

const chunkingConfig = {
    minLength: 1000,
    maxLength: 2000,
    splitter: 'sentence',
    overlap: 0,  // number of overlap chracters
    delimiters: '', // regex for base split method
  } as any;

const pdfIndexer = devLocalIndexerRef('menuQA')  
const devLocalRetriever = devLocalRetrieverRef('menuQA')

const hybridReranker = ai.defineReranker(
    {
      name: 'custom/reranker',
      configSchema: z.object({
        k: z.number().optional(),
        alpha: z.number().optional(),
      }),
    },
    async (query, documents, options) => {

        const queryText = query.text?.toLowerCase() || '';
        logger.info(`Re-ranker received query: ${queryText}`);

        const alpha = options.alpha ?? 0.5

        // --- Compute final hybrid score and sort ---
        const merged = Array.from(documents)
        .map((doc) => {
            const hybridScore = alpha * doc.metadata?.denseScore + (1 - alpha) * doc.metadata?.sparseScore;
            return {
                    ...doc,
                    metadata: {
                        ...doc.metadata,
                        score: hybridScore
                    },
                };
        });

        const topK = merged
            .sort((a, b) => (b.metadata?.score ?? 0) - (a.metadata?.score ?? 0))
            .slice(0, options.k || 3);

        return { documents: topK }; 
  });
 
const keywordScoreRetriever = ai.defineRetriever({
        name: 'custom/sparseRetriever',
        info: { label: 'Sparse (Keyword Scored) Retriever' },
        },
        async (query: Document, options: z.infer<typeof CommonRetrieverOptionsSchema>) => {
            logger.info(`Sparse Retriever received query: ${query.text}`);

            const k = options?.k ?? 10;

            // const docs = await bm25KeywordMatch(query, options?.k ?? 10);
            const allDocs = await ai.retrieve({
                retriever: devLocalRetriever,
                query,
                options: { k }
            });
            const queryKeywords = new Set(query.text?.split(/\s+/));

            const matchDocs = allDocs
                .map((doc) => {
                const text = doc.text || '';
    
                const matchScore = Array.from(queryKeywords).reduce((acc, word) => acc + (text.includes(word) ? 1 : 0), 0);
    
                // Create a valid Document instance
                // TODO: use spray operator here, i.g. return { ...doc, id: <>, matchScore: <>}
                return Document.fromText(text, 
                    {
                        ...(doc.metadata || {}),
                        id: getHash(text),
                        matchScore,
                    });
            })
            .sort((a, b) => (b.metadata?.matchScore ?? 0) - (a.metadata?.matchScore ?? 0))
            .slice(0, k);

            return {
                documents: matchDocs
            }
        }
)

const hybridRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
    // 'k' is already in CommonRetrieverOptionsSchema, but you could add others:
    preRerankK: z.number().max(1000).optional().describe("Number of documents to retrieve before potential reranking"),
    customFilter: z.string().optional().describe("A custom filter string"),
    alpha: z.number()
});

const hybridRetriever = ai.defineRetriever({
        name: 'custom/hybridRetriever',
        info: { label: 'Hybrid Retriever (dense + sparse)' },
        configSchema: hybridRetrieverOptionsSchema,
    }, async( query: Document, options: z.infer<typeof hybridRetrieverOptionsSchema> ) => {
        
        logger.info(`Hybrid Retriever received query: ${query.text}`);

        const initialK = options.preRerankK || 10; // Default to 10 if not provided
        const finalK = options.k ?? 3; // Default final number of docs to 3 if k is not set
        const alpha = options.alpha ?? 0.5; // balance: 0.5 = equal weight
        
        // --- Merge & Deduplicate Results ---
        // Use a Map to store unique documents by content hash
        // Assign scores from both retrieval methods.        
        const allDocsMap = new Map<string, 
            { 
                doc: Document; 
                denseScore?: number; 
                sparseScore?: number 
           }>();

        // --- Dense retrieval from vector store
        const denseDocs = await ai.retrieve({ // kNN or ANN inside
            retriever: devLocalRetriever,
            query: query,
            options: { k: initialK }
        });
        denseDocs.forEach((doc, i) => {
            const docHash = getHash(doc.text);
            // Assign a simple rank-based score (higher rank = higher score)
            allDocsMap.set(docHash, { doc, denseScore: initialK - i });
        });        

        // --- Sparse retrieval
        const sparseDocs = await ai.retrieve({
            retriever: keywordScoreRetriever,
            query: query,
            options: { k: initialK }
        });
        sparseDocs.forEach((doc, i) => {
                const docHash = getHash(doc.text);
                const existingDoc = allDocsMap.get(docHash);
                if (existingDoc) {
                    existingDoc.sparseScore = initialK - i;
                } else {
                    allDocsMap.set(docHash, { doc, sparseScore: initialK - i });
                }
        });

        const combinedDocsWithScores = Array.from(allDocsMap.values())
            .map(({ doc, denseScore, sparseScore }) => 
            {
                doc.metadata = {
                    ...doc.metadata || {},
                    denseScore: denseScore ?? 0,
                    sparseScore: sparseScore ?? 0,
                }
                return doc;
            });

        const rerankedDocs = await ai.rerank({
            reranker: hybridReranker,
            query: query,
            documents: combinedDocsWithScores,
            options:  { k: finalK, alpha: alpha }
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
            return Document.fromText(text, 
                { 
                    id: getHash(text),
                    filePath 
                });
        });

        // Add documents to the index.
        const index = await ai.index({
            indexer: pdfIndexer,
            documents
        });
    }
);

//
// Tools can be used on ai.prompt just like in ai.generate()
//
export const promptFlow = ai.defineFlow(
    {
        name: "promptFlow",
        inputSchema: z.string().describe("Prompt input"),
    },
    async (input: string) => {

        // retrieve relevant documents. Uses kNN internally, then re-ranks the retrieved docs
        const docs = await ai.retrieve({
            retriever: hybridRetriever, //use the custom retriever
            query: input,
            options: {
                k: 3,
                preRerankK: 10,
                alpha: 0.7,
                customFilter: "words count > 5",
            }
        });

        // This is Dotprompt (see https://github.com/google/dotprompt)
        const prompt = ai.prompt('tools_agent'); // '.prompt' extension will be added automatically
        const promptResponse = await prompt({ // Full conversation turn with LLM here. After it completes, 'text' contains the response text
                // Prompt input
                input,
                docs: docs
            },
            {
                tools: [getEvents]
            }
        ); 
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
            retriever: hybridRetriever, //use the custom retriever
            query: input,
            options: {
                k: 3,
                preRerankK: 10,
                alpha: 0.7,
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

startFlowServer({
    flows: [RAGFlow, indexFlow, promptFlow],
    port: 3400,
});