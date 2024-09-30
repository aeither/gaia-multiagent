import { createOpenAI as createGroq } from "@ai-sdk/openai";
import { generateText, tool } from "ai";
import dotenv from "dotenv";
import { z } from "zod";
dotenv.config();

const groq = createGroq({
	baseURL: "https://llamatool.us.gaianet.network/v1",
	apiKey: "no_key",
});

// Simulated weather API function
async function getWeather(
	location: string,
): Promise<{ temperature: number; condition: string }> {
	const randomTemp = Math.floor(Math.random() * 30) + 10;
	const conditions = ["Sunny", "Cloudy", "Rainy", "Windy"];
	const randomCondition =
		conditions[Math.floor(Math.random() * conditions.length)];
	return { temperature: randomTemp, condition: randomCondition };
}

// Simulated local time API function
async function getLocalTime(location: string): Promise<string> {
	const offset = Math.floor(Math.random() * 24) - 12; // Random UTC offset between -12 and +12
	const now = new Date();
	now.setHours(now.getHours() + offset);
	return now.toLocaleTimeString("en-US", {
		hour: "2-digit",
		minute: "2-digit",
	});
}

async function main() {
	const { text } = await generateText({
		model: groq("llama"),
		tools: {
			getWeather: tool({
				description: "Get the current weather for a location",
				parameters: z.object({
					location: z
						.string()
						.describe("The city and country to get weather for"),
				}),
				execute: async ({ location }) => {
					const weatherData = await getWeather(location);
					return `The weather in ${location} is ${weatherData.condition} with a temperature of ${weatherData.temperature}°C`;
				},
			}),
			getLocalTime: tool({
				description: "Get the current local time for a location",
				parameters: z.object({
					location: z
						.string()
						.describe("The city and country to get local time for"),
				}),
				execute: async ({ location }) => {
					const localTime = await getLocalTime(location);
					return `The current local time in ${location} is ${localTime}`;
				},
			}),
		},
		maxSteps: 5,
		prompt: "What's the weather and local time in Tokyo, Japan?",
	});

	console.log("Assistant:", text);
}

main().catch(console.error);