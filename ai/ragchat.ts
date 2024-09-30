import { RAGChat, custom } from "@upstash/rag-chat";
import { Index } from "@upstash/vector";
import dotenv from "dotenv";

dotenv.config();

if (!process.env.UPSTASH_VECTOR_REST_URL)
	throw new Error("UPSTASH_VECTOR_REST_URL not found");
if (!process.env.UPSTASH_VECTOR_REST_TOKEN)
	throw new Error("UPSTASH_VECTOR_REST_TOKEN not found");

const UPSTASH_VECTOR_REST_URL = process.env.UPSTASH_VECTOR_REST_URL;
const UPSTASH_VECTOR_REST_TOKEN = process.env.UPSTASH_VECTOR_REST_TOKEN;

// Initialize the Vector Index
const vectorIndex = new Index({
	url: UPSTASH_VECTOR_REST_URL,
	token: UPSTASH_VECTOR_REST_TOKEN,
});

// Initialize RAGChat with custom LLaMA Tool model
const ragChat = new RAGChat({
	model: custom("phi", {
		apiKey: "no_key",
		baseUrl: "https://phi.us.gaianet.network/v1",
	}),
	vector: vectorIndex,
});

function createMockText(): string {
	return `
    Bill Evans was an American jazz pianist and composer who is widely considered to be one of the most influential figures in the history of jazz piano. Born in 1929 in Plainfield, New Jersey, Evans began playing piano at a young age and quickly showed a natural talent for the instrument.

    Evans studied classical piano and composition at Southeastern Louisiana University, where he graduated in 1950. After a brief stint in the Army, he moved to New York City to pursue a career in jazz. In the late 1950s, Evans joined the Miles Davis Sextet, where he played on the landmark album "Kind of Blue."

    One of Evans' most famous albums is "Waltz for Debby," recorded live at the Village Vanguard in New York City in 1961. The album features Evans' trio with bassist Scott LaFaro and drummer Paul Motian. The title track, "Waltz for Debby," was written by Evans for his niece Debby.

    Throughout his career, Evans developed a unique and influential style of jazz piano playing characterized by his use of impressionistic harmony, introspective melodies, and a light, "singing" touch. He was known for his ability to create complex harmonies and his use of block chords, which became a hallmark of his style.

    Evans struggled with drug addiction for much of his life, which affected his health and career. Despite this, he continued to perform and record until his death in 1980 at the age of 51. His legacy continues to influence jazz pianists and musicians to this day.
  `;
}

async function main() {
	const mockText = createMockText();

	// Add the mock text to the RAGChat context
	await ragChat.context.add({
		type: "text",
		data: mockText,
	});

	// Example chat interaction
	const response = await ragChat.chat(
		"Tell me about Bill Evans' most famous album.",
	);
	console.log("AI Response:", response.output);

	// Retrieve chat history
	const history = await ragChat.history.getMessages({ amount: 2 });
	console.log("Chat History:", history);
}

main().catch(console.error);