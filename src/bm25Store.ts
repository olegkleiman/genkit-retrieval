const bm25 = require('wink-bm25-text-search'); // no types for this module available
const nlp = require( 'wink-nlp-utils' );
import { logger } from 'genkit/logging';
import fs from 'fs';
import { DocumentCacheEntry, setDocumentInCache, getDocumentFromCache } from './redisCache'

export interface BMScoredDocument extends DocumentCacheEntry {
    score: number
}

export class BM25Engine {

    private static instance: BM25Engine;
    private engine: any;
    private isIndexLoaded: boolean = false;

    private pipe = [
        nlp.string.lowerCase,
        nlp.string.tokenize0,
        nlp.tokens.removeWords,
        nlp.tokens.stem,
        nlp.tokens.propagateNegations
      ]; 

    private constructor() {
        this.engine = bm25();
    }

    private applyConfigAndPrepTasks() {
        // Apply configuration and NLP pipeline
        // This should be called before adding documents or searching on a fresh/imported engine
        const config = this.engine.getConfig();
        if( !config ) {
            this.engine.defineConfig({ fldWeights: { text: 1 } }); 
        }
        this.engine.definePrepTasks(this.pipe);
    }

    async buildIndex(docs: DocumentCacheEntry[], storePath: string) {

        this.applyConfigAndPrepTasks()

        for (const [i, doc] of docs.entries()) {
            // Note, 'i' becomes the unique id for 'doc'
            this.engine.addDoc({ text: doc.text }, i);
            try {
                await setDocumentInCache(i, doc);
            } catch(error) {
                logger.error(`Error setting document in Redis cache for docId ${i}:`, error);
            }
        }

        this.engine.consolidate();
        logger.info('BM25 index consolidation complete.');

        const indexData = this.engine.exportJSON();
        fs.writeFileSync(storePath, JSON.stringify(indexData));
    }

    loadIndexFromStore(storePath: string): boolean {
        if( this.isIndexLoaded )
            return true;

        if (!fs.existsSync(storePath)) {
            logger.warn(`BM25 index file not found at ${storePath}. Engine will not be loaded.`);
            this.isIndexLoaded = false;
            return false;
        }
    
        try
        {
            const savedData = JSON.parse(fs.readFileSync(storePath, 'utf8'));
            this.engine.importJSON(savedData);
            // Crucial: Re-apply config and prep tasks after importing
            this.applyConfigAndPrepTasks();

            this.isIndexLoaded = true;
            return true;
        } catch(error) {
            this.isIndexLoaded = false;
            return false;
        }
    }

    async search(query: string, limit: number = 10): Promise<BMScoredDocument[]> {

        if (!this.isIndexLoaded) {
            logger.warn('BM25Engine search called, but index is not loaded. Returning empty results.');
            return [];
        }

        // The engine.search() method returns [docInternalIndex, score].
        // The docInternalIndex is simply the 0-based order in which documents were added during engine.addDoc(doc, i).
        const results = this.engine.search(query, limit);

        // Use Promise.all to handle asynchronous operations
        const resolvedResults = await Promise.all(
            results.map(async (res: [string | number, string | number]) => {

                const docId = parseInt(String(res[0]), 10);
                const score = parseFloat(String(res[1]));

                const cachedDoc: DocumentCacheEntry | null = await getDocumentFromCache(docId);
                if( cachedDoc ) {
                    return { ...cachedDoc, score }
                }

                return null;
            })
        );

        // Filter out nulls if a document was not found in cache
        return resolvedResults.filter(doc => doc !== null) as BMScoredDocument[];

    }

    static getInstance() : BM25Engine {
        if( !BM25Engine.instance )
            BM25Engine.instance = new BM25Engine();

        return BM25Engine.instance;
    }

}
