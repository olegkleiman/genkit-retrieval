import { createClient, RedisClientType } from 'redis';
import { logger } from 'genkit/logging';
import dotenv from 'dotenv';
dotenv.config();

const redisUrl = `redis://${process.env.REDIS_HOST || '127.0.0.1'}:${process.env.REDIS_PORT || '6379'}`;
let redisClient: RedisClientType;  

export interface DocumentCacheEntry {
    text: string;
    originalFilePath: string;
}

(async () => {

    redisClient = createClient({
        url: redisUrl,
        password: process.env.REDIS_PASSWORD || undefined, // Password should be undefined if not set
    });

    redisClient.on('error', (err) => logger.error('Redis Client Error', err));
    redisClient.on('connect', () => logger.info('Connecting to Redis...'));
    redisClient.on('ready', () => logger.info('Connected to Redis and ready to use.'));
  
    try {
        await redisClient.connect();
    } catch (err) {
        logger.error('Failed to connect to Redis:', err);
    }
})();

//const CACHE_EXPIRATION_SECONDS = 3600; // 1 hour, adjust as needed

export async function getDocumentFromCache(docId: number): Promise<DocumentCacheEntry | null> {
    try {

        if (!redisClient || !redisClient.isReady) {
            logger.warn('Redis client not ready, skipping cache get.');
            return null;
        }

        const cachedData = await redisClient.json.get(`doc:${docId}`);
        // const cachedData = await redisClient.get(`doc:${docId}`);
        if (cachedData) {
          return typeof cachedData === 'string' ? JSON.parse(cachedData) as DocumentCacheEntry : null;
        }

        logger.debug(`Cache MISS for docId: ${docId}`);
        return null;

    } catch (error ) {
        logger.error(`Error getting document from Redis cache for docId ${docId}:`, error);
        return null; // Treat cache errors as a cache miss
    }
}

export async function setDocumentInCache(docId: number, documentData: DocumentCacheEntry): Promise<void> {
    try {

        if (!redisClient || !redisClient.isReady) {
            logger.warn('Redis client not ready, skipping cache set.');
            return;
        }
        await redisClient.json.set(`doc:${docId}`, '$', { ...documentData });

    } catch (error) {
      logger.error(`Error setting document in Redis cache for docId ${docId}:`, error);
    }
  }