# Genkit Flows - Hybrid Search & RAG

Hybrid Search flow is implemented as a combination of BM25 (sparse) retrival and embeddings (dense) retrieval approaches. The results are merged using RRF algorithm in Genkit re-ranker.
RAG Flow invokes the Search Flow explicicly grounding the search to the returned docs.
You may run Genkit developer UI to try the both.
```
npx genkit start -- npm run dev 
```

### 1. You might presumable need Genkit CLI. 
Install from [here](https://github.com/firebase/genkit) and check the installation with
```
genkit --version
```
### 2. Create GEMINI_API_KEY
from [here](https://aistudio.google.com/app/apikey)
and create .env file with 
```
GEMINI_API_KEY=<API key>
```

### 3. Get access to Redis with JSON module
update the .env configuration with corresponding host, port and password settings 