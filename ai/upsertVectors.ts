import { createOpenAI } from "@ai-sdk/openai";
import { Index } from "@upstash/vector";
import { embedMany } from "ai";
import dotenv from "dotenv";

dotenv.config();

if (!process.env.UPSTASH_VECTOR_REST_URL)
	throw new Error("UPSTASH_VECTOR_REST_URL not found");
const UPSTASH_VECTOR_REST_URL = process.env.UPSTASH_VECTOR_REST_URL;
if (!process.env.UPSTASH_VECTOR_REST_TOKEN)
	throw new Error("UPSTASH_VECTOR_REST_TOKEN not found");

const UPSTASH_VECTOR_REST_TOKEN = process.env.UPSTASH_VECTOR_REST_TOKEN;
const index = new Index({
	url: UPSTASH_VECTOR_REST_URL,
	token: UPSTASH_VECTOR_REST_TOKEN,
});

const groq = createOpenAI({
	baseURL: "https://llamatool.us.gaianet.network/v1",
	apiKey: "no_key",
});

async function getEmbeddings(chunks: string[]): Promise<number[][]> {
	const cleanChunks = chunks.map((c) => c.replace("\n", " "));
	const { embeddings } = await embedMany({
		model: groq.embedding("bge-base-en-v1.5"),
		values: cleanChunks,
	});
	return embeddings;
}

function createMockText(): string {
	return `
    Bill Evans was an American jazz pianist and composer who is widely considered to be one of the most influential figures in the history of jazz piano. Born in 1929 in Plainfield, New Jersey, Evans began playing piano at a young age and quickly showed a natural talent for the instrument.

    Evans studied classical piano and composition at Southeastern Louisiana University, where he graduated in 1950. After a brief stint in the Army, he moved to New York City to pursue a career in jazz. In the late 1950s, Evans joined the Miles Davis Sextet, where he played on the landmark album "Kind of Blue."

    One of Evans' most famous albums is "Waltz for Debby," recorded live at the Village Vanguard in New York City in 1961. The album features Evans' trio with bassist Scott LaFaro and drummer Paul Motian. The title track, "Waltz for Debby," was written by Evans for his niece Debby.

    Throughout his career, Evans developed a unique and influential style of jazz piano playing characterized by his use of impressionistic harmony, introspective melodies, and a light, "singing" touch. He was known for his ability to create complex harmonies and his use of block chords, which became a hallmark of his style.

    Evans struggled with drug addiction for much of his life, which affected his health and career. Despite this, he continued to perform and record until his death in 1980 at the age of 51. His legacy continues to influence jazz pianists and musicians to this day.
  `;
}

function chunkText(text: string): string[] {
	return text.split(/\n+/).filter((chunk) => chunk.trim().length > 0);
}

async function upsertVectors(chunks: string[]): Promise<void> {
	const embeddings = await getEmbeddings(chunks);
	const vectors = chunks.map((chunk, i) => ({
		id: `chunk-${i}`,
		vector: embeddings[i],
		metadata: { text: chunk },
	}));
	await index.upsert(vectors);
	console.log(`${vectors.length} vectors upserted successfully`);
}

async function main() {
	const mockText = createMockText();
	const chunks = chunkText(mockText);
	await upsertVectors(chunks);
}

main().catch(console.error);
