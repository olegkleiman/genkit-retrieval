import { genkit, z } from 'genkit/beta';
import { googleAI, gemini20Flash, gemini25FlashPreview0417 } from '@genkit-ai/googleai';
import { textEmbedding004, vertexAI } from '@genkit-ai/vertexai';
import { 
    devLocalIndexerRef, 
    devLocalRetrieverRef,
    devLocalVectorstore
 } from '@genkit-ai/dev-local-vectorstore';

export const ai = genkit({

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