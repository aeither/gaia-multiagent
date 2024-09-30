import { createOpenAI } from "@ai-sdk/openai";
import { Index } from "@upstash/vector";
import { embed, generateText, tool } from "ai";
import dotenv from "dotenv";
import { z } from "zod";
import {
	MKBHD_PROMPT,
	MR_WHOS_THE_BOSS_PROMPT,
	UNBOX_THERAPY_PROMPT,
} from "./systemPrompts";

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
const inventory = createOpenAI({
	baseURL: "https://llamatool.us.gaianet.network/v1",
	apiKey: "no_key",
});

const moderator = createOpenAI({
	baseURL: "https://sreeram.us.gaianet.network/v1",
	apiKey: "no_key",
});

const llama = createOpenAI({
	baseURL: "https://llama.us.gaianet.network/v1",
	apiKey: "no_key",
});

const phi = createOpenAI({
	baseURL: "https://phi.us.gaianet.network/v1",
	apiKey: "no_key",
});

const gemma = createOpenAI({
	baseURL: "https://gemma.us.gaianet.network/v1",
	apiKey: "no_key",
});

async function getEmbedding(text: string): Promise<number[]> {
	const { embedding } = await embed({
		model: inventory.embedding("bge-base-en-v1.5"),
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
	const generateInventoryText = async (prompt: string) => {
		const { text } = await generateText({
			model: inventory("llama"),
			tools: { billEvansQA },
			maxSteps: 5,
			prompt,
			// onStepFinish({ text, toolCalls, toolResults, finishReason, usage }) {
			// 	console.log('text:', text);
			// 	console.log('toolCalls:', toolCalls);
			// 	console.log('toolResults:', toolResults);
			// 	console.log('finishReason:', finishReason);
			// 	console.log('usage:', usage);
			// },
		});
		return text;
	};

	const generateMKBHDText = async (prompt: string) => {
		const { text } = await generateText({
			model: llama("llama"),
			maxSteps: 5,
			system: MKBHD_PROMPT,
			prompt,
		});
		return text;
	};

	const generateUnboxTherapyText = async (prompt: string) => {
		const { text } = await generateText({
			model: phi("phi"),
			maxSteps: 5,
			system: UNBOX_THERAPY_PROMPT,
			prompt,
		});
		return text;
	};

	const generateMrWhosTheBossText = async (prompt: string) => {
		const { text } = await generateText({
			model: gemma("gemma"),
			maxSteps: 5,
			system: MR_WHOS_THE_BOSS_PROMPT,
			prompt,
		});
		return text;
	};

	const generateModeratorSummary = async (
		mkbhdText: string,
		unboxTherapyText: string,
		mrWhosTheBossText: string,
	) => {
		const prompt = `Summarize and compare the following reviews of the iPhone 15 Pro from three tech reviewers. Then provide a conclusion to help with the buying decision:

MKBHD Review:
${mkbhdText}

Unbox Therapy Review:
${unboxTherapyText}

MrWhosTheBoss Review:
${mrWhosTheBossText}

Please provide a summary that compares the key points from each reviewer, highlighting agreements and disagreements. Then, offer a conclusion to help potential buyers make an informed decision.`;

		const { text } = await generateText({
			model: moderator("llama"),
			maxSteps: 5,
			system:
				"You are an impartial moderator tasked with summarizing and comparing tech reviews. Provide clear, concise summaries and a balanced conclusion to aid in purchasing decisions.",
			prompt,
		});
		return text;
	};

	const prompt =
		"What are your thoughts on the latest iPhone 16 Pro? Compare its features, performance, and value proposition to its main Android competitors. Also, how does its camera system stack up against other flagship phones?";

	const inventoryText = await generateInventoryText(prompt);
	const mkbhdText = await generateMKBHDText(prompt);
	const unboxTherapyText = await generateUnboxTherapyText(prompt);
	const mrWhosTheBossText = await generateMrWhosTheBossText(prompt);

	console.log("Inventory:", inventoryText);
	console.log("MKBHD:", mkbhdText);
	console.log("Unbox Therapy:", unboxTherapyText);
	console.log("MrWhosTheBoss:", mrWhosTheBossText);

	// Generate and log the moderator's summary
	const moderatorSummary = await generateModeratorSummary(
		mkbhdText,
		unboxTherapyText,
		mrWhosTheBossText,
	);
	console.log("\nModerator's Summary and Conclusion:");
	console.log(moderatorSummary);
}

main().catch(console.error);
