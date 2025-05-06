// from https://firebase.google.com/docs/genkit/rag

import crypto from 'crypto';
import { googleAI } from '@genkit-ai/googleai';

import { z } from 'genkit/beta';

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
import { Document, CommonRetrieverOptionsSchema } from 'genkit/retriever';
// import { RankedDocument, CommonRerankerOptionsSchema } from 'genkit/reranker';
import { logger } from 'genkit/logging';
import { BM25Engine} from './bm25Store';
import { chunk } from 'llm-chunk';
import { startFlowServer } from '@genkit-ai/express';
import path from 'path';
import { readFile } from 'fs/promises';
import pdf from 'pdf-parse';
import fs from 'fs';
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

const bm25EngineInstance = BM25Engine.getInstance();

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

// The fundamental purpose of a re-ranker is to re-order the list of documents
// based on their relevance to the specific query.
// In the case of RRF-ranking, however, the re-ordiring is based on pre-calculated ranks
// and, hence, 'query' parameter is not used
const hybridReranker = ai.defineReranker(
    {
        name: "custom/reranker"
    },
    async(query, documents, options) => {

        // --- Reciprocal Rank Fusion (RRF) ---
        // see here: https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a

        const rrfConstant = 60; // Common value for k in RRF

        const fusedDocs = documents.map(doc => {

            let rrfScore = 0;

            // Add score based on dense rank (lower rank is better)
            rrfScore += 1 / (rrfConstant + doc.metadata?.denseRank);
            // Add score based on sparse rank (lower rank is better)
            rrfScore += 1 / (rrfConstant + doc.metadata?.sparseRank);            

            return {
                ...doc,
                metadata: {
                    score: rrfScore,
                    ...doc.metadata
                }
            };
        })

        const topK = fusedDocs
            .sort((a, b) => (b.metadata.score ?? 0) - (a.metadata.score ?? 0))
            .slice(0, options?.k || 3);

        return {
            documents: topK 
        }
    }
)

const bm25Retriever = ai.defineRetriever(
    {
        name: 'custom/bm25Retriever',
        info: { label: 'Sparse (BM25) Retriever' },
    },
    async(query: Document, options: z.infer<typeof CommonRetrieverOptionsSchema>) => {
        logger.info(`Sparse Retriever received query: ${query.text}`);

        bm25EngineInstance.fromStore('bm25-index.json');
        
        const results = await bm25EngineInstance.search(query.text)

        const docs = results.map( res => {
            return Document.fromText(res.text, 
                        {
                            score: res.score
                        }
            )
        })

        return {
            documents: docs
        }
    }
)

const keywordScoreRetriever = ai.defineRetriever({
        name: 'custom/sparseRetriever',
        info: { label: 'Sparse (Keyword Scored) Retriever' },
        },
        async (query: Document, options: z.infer<typeof CommonRetrieverOptionsSchema>) => {
            logger.info(`Sparse Retriever received query: ${query.text}`);

            const k = options?.k ?? 10;

            // Fetch initial dense results
            const allDocs = await ai.retrieve({
                retriever: devLocalRetriever,
                query,
                options: { k }
            });

            const queryKeywords = new Set(query.text?.split(/\s+/));

            const matchDocs = allDocs
                .map((doc) => {
                const text = doc.text || '';
    
                // Calculate simple keyword overlap score
                const score = Array.from(queryKeywords).reduce((acc, word) => acc + (text.includes(word) ? 1 : 0), 0);
    
                // Create a valid Document instance
                // TODO: use spray operator here, i.g. return { ...doc, id: <>, matchScore: <>}
                return Document.fromText(text, 
                    {
                        ...(doc.metadata || {}),
                        id: getHash(text),
                        score,
                    });
            })
            .sort((a, b) => (b.metadata?.score ?? 0) - (a.metadata?.score ?? 0))
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
});

const hybridRetriever = ai.defineRetriever({
        name: 'custom/hybridRetriever',
        info: { label: 'Hybrid Retriever (dense + sparse)' },
        configSchema: hybridRetrieverOptionsSchema,
    }, async( query: Document, options: z.infer<typeof hybridRetrieverOptionsSchema> ) => {
        
        logger.info(`Hybrid Retriever received query: ${query.text}`);

        const initialK = options.preRerankK || 10; // Default to 10 if not provided
        const finalK = options.k ?? 3; // Default final number of docs to 3 if k is not set
        
        // --- Merge & Deduplicate Results ---
        // Use a Map to store unique documents by content hash
        // Assign scores from both retrieval methods.        
        const allDocsMap = new Map<string, 
            { 
                doc: Document; 
                denseRank?: number; 
                sparseRank?: number 
           }>();

        // --- Dense retrieval from vector store
        const denseDocs = await ai.retrieve({ // kNN or ANN inside
            retriever: devLocalRetriever,
            query: query,
            options: { k: initialK }
        });
        // Actual relevance scores aren't readily available
        denseDocs.forEach((doc, i) => {
            const docHash = getHash(doc.text);
            // Assign a simple rank-based score (higher rank = higher score)
            allDocsMap.set(docHash, { doc, denseRank: i });
        });        

        // --- Sparse retrieval
        const sparseDocs = await ai.retrieve({
            retriever: bm25Retriever, // keywordScoreRetriever,
            query: query,
            options: { k: initialK }
        });
        sparseDocs.forEach((doc, i) => {
                const docHash = getHash(doc.text);
                const existingDoc = allDocsMap.get(docHash);
                if (existingDoc) {
                    existingDoc.sparseRank = i;
                } else {
                    allDocsMap.set(docHash, { doc, sparseRank: i });
                }
        });

        const combinedDocsWithScores = Array.from(allDocsMap.values())
            .map(({ doc, denseRank, sparseRank }) => 
            {
                doc.metadata = {
                    ...doc.metadata || {},
                    denseRank: denseRank ?? 0,
                    sparseRank: sparseRank ?? 0,
                }
                return doc;
            });

        const rerankedDocs = await ai.rerank({
            reranker: hybridReranker,
            query: query,
            documents: combinedDocsWithScores,
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
            return Document.fromText(text, 
                { 
                    id: getHash(text),
                    filePath 
                });
        });

        // Actually generate embedding of each document 
        // an store this embedding (along with doc's metadata) into 'devLocalVestorStore'
        await ai.index({
            indexer: pdfIndexer,
            documents
        });

        const bm25Docs = chunks.map( _chunk => {
            return {
                id: getHash(_chunk),
                text: _chunk,
                originalFilePath: filePath
            }
        });
        ai.run('buid-bm25-index', async () => {
            bm25EngineInstance.buildIndex(bm25Docs, 'bm25-index.json');
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

export const SearchFlow = ai.defineFlow(
    {
        name: "SearchFlow",
        inputSchema: z.string(),        
    },
    async (input: string) => {

        // retrieve relevant documents. Uses kNN internally, then re-rank the retrieved docs
        const docs = await ai.retrieve({
            retriever: hybridRetriever, //use the custom retriever
            query: input,
            options: {
                k: 3,
                preRerankK: 10,
                customFilter: "words count > 5",
            }
        });

        return docs;
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

        // Run SearchFlow - RAG step
        const docs = await SearchFlow(input);

        // generate a response
        const llmResponse =  ai.generate({
            tools: [getEvents],
            // returnToolRequests: true,
            prompt: `
                You are an AI assistant. Answer the following question based ONLY on the provided context documents.
                If the answer is not found in the documents, state that you cannot answer based on the provided context.

                Context Documents:\n---\n{{#docs}}{{content.text}}\n---\n{{/docs}}
                Question: ${input}`,
            // prompt: `
            //     Question: ${input}`,                
            docs
        });

        const toolRequests = (await llmResponse).toolRequests;
        console.log("Tool requests: ", toolRequests);

        return llmResponse;
    }
);

startFlowServer({
    flows: [SearchFlow, RAGFlow, indexFlow, promptFlow],
    port: 3400,
});