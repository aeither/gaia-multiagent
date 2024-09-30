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

// function createMockText(): string {
// 	return `
//     key points about the iPhone 16 Pro:
//     Key Features
//     Built for Apple Intelligence, a new AI system for writing, productivity, and communication
//     A18 Pro chip with improved Neural Engine, CPU, and GPU performance
//     New 48MP Ultra Wide camera in addition to existing 48MP main camera
//     5x optical zoom Telephoto camera on both Pro models
//     Camera Control for easier access to camera tools and settings
//     4K 120fps Dolby Vision video recording
//     Four studio-quality mics for improved audio recording
//     Larger 6.3" and 6.9" displays with thinner borders
//     Titanium design that's stronger and lighter
//     Up to 4 more hours of battery life compared to previous models
//     Other Highlights
//     New Desert Titanium color option
//     iOS 18 with more customization options
//     Satellite connectivity for emergency services and messaging
//     Improved gaming performance with ray tracing
//     MagSafe charging up to 25W
//     Focus on privacy and security features
//     Environmental initiatives like increased use of recycled materials
//     The iPhone 16 Pro seems to focus on AI capabilities, camera improvements, performance enhancements, and design refinements compared to previous models.
//   `;
// }
// function createMockText(): string {
// 	return `
//     key points about the Google Pixel 9:
//     Main Specifications
//     Starting price: €899 or €299.67 per month for 3 months
//     Available colors: Peony Pink, Matcha Green, Clay Gray, Obsidian Black
//     6.3-inch Actua display with up to 120 Hz refresh rate
//     Google Tensor G4 chip with 12 GB RAM
//     Battery life of over 24 hours (up to 100 hours in extreme battery saver mode)
//     7 years of OS and security updates
//     Camera and AI Features
//     50 MP main camera with Night Sight mode
//     48 MP ultrawide camera with Macro mode
//     High Definition Zoom up to 8x
//     10.5 MP front camera with autofocus
//     AI features like Magic Editor for advanced photo editing
//     Gemini, integrated AI assistant
//     Design and Security
//     Damage-resistant front and back glass
//     Curved edges and durable frame
//     Water and dust resistant (IP68 rated)
//     Security features like Crisis Alerts and crash detection
//     Other Details
//     Dual SIM (physical + eSIM)
//     5G network compatible
//     Includes 6 months of Google One Premium (2 TB storage)
//     3 months of YouTube Premium included
//     6 months of Fitbit Premium included
//     The Pixel 9 is positioned as a high-end smartphone with a strong focus on camera capabilities and AI integration in everyday features.
//   `;
// }

function createMockText(): string {
	return `
    key points about the Galaxy S24 and S24+:
    New AI Features (Galaxy AI):
    • Circle to Search: Allows circling objects on screen to initiate Google searches
    • Live Translate: Translates calls in real-time
    • Interpreter: Translates face-to-face conversations
    • Chat Assist: Helps write messages in different tones and languages
    • Generative Edit: Allows editing and generating parts of images with AI
    Improved Hardware:
    • Brighter and larger displays (6.2" FHD+ on S24, 6.7" QHD+ on S24+)
    • More powerful processor with improved NPU for AI
    • Larger vapor chamber for better heat dissipation
    • Larger batteries
    Camera:
    • 50MP main sensor
    • AI Zoom to enhance quality of zoomed photos
    • Improved Nightography for low-light photos
    Design:
    • Thinner bezels
    • New colors (Onyx Black, Marble Gray, Cobalt Violet, Amber Yellow)
    Other Features:
    • Up to 512GB storage
    • Ray tracing support for games
    • New lock screen widgets
    The Galaxy S24 and S24+ are positioned as high-end smartphones with a strong focus on new AI capabilities that leverage the enhanced hardware to offer new experiences to users.
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
