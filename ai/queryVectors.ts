import { createOpenAI as createGroq } from "@ai-sdk/openai";
import { Index } from "@upstash/vector";
import { embed, generateText, tool } from "ai";
import dotenv from "dotenv";
import { z } from "zod";

dotenv.config();

if (!process.env.UPSTASH_VECTOR_URL)
	throw new Error("UPSTASH_VECTOR_URL not found");
const UPSTASH_VECTOR_URL = process.env.UPSTASH_VECTOR_URL;
if (!process.env.UPSTASH_VECTOR_REST_TOKEN)
	throw new Error("UPSTASH_VECTOR_REST_TOKEN not found");

const UPSTASH_VECTOR_REST_TOKEN = process.env.UPSTASH_VECTOR_REST_TOKEN;
const index = new Index({
	url: UPSTASH_VECTOR_URL,
	token: UPSTASH_VECTOR_REST_TOKEN,
});

const groq = createGroq({
	baseURL: "https://llamatool.us.gaianet.network/v1",
	apiKey: "no_key",
});

async function getEmbedding(text: string): Promise<number[]> {
	const { embedding } = await embed({
		model: groq.embedding("bge-base-en-v1.5"),
		value: text.replace("\n", " "),
	});
	return embedding;
}

async function askQuestion(question: string): Promise<string> {
	const questionEmbedding = await getEmbedding(question);
	const results = await index.query({
		vector: questionEmbedding,
		topK: 3,
		includeMetadata: true,
	});
	const context = results.map((r) => r.metadata?.text ?? "").join("\n");

	const prompt = `Question: ${question}\n\nContext: ${context}\n\nAnswer:`;
	return prompt;
}

const billEvansQA = tool({
	description:
		"Answer questions about Bill Evans based on the provided context",
	parameters: z.object({
		question: z
			.string()
			.describe("The question about Bill Evans to be answered"),
	}),
	execute: async ({ question }) => {
		const answer = await askQuestion(question);
		return `Question: ${question}\nAnswer: ${answer}`;
	},
});

async function main() {
	const { text } = await generateText({
		model: groq("llama"),
		tools: { billEvansQA },
		maxSteps: 5,
		prompt:
			"Who is Debby in the album 'Waltz for Debby'? Also, where did Bill Evans study?",
	});

	console.log("Assistant:", text);
}

main().catch(console.error);