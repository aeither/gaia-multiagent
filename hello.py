import logging
import os
import sys

from crewai import Agent, Crew, Process, Task
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

duckduckgo_search = DuckDuckGoSearchRun()

def create_travel_crew(destination):
    # Define Agents
    travel_advisor = Agent(
        role="Travel Advisor",
        goal=f"Craft a personalized itinerary for a trip based on your preferences.",
        backstory="A seasoned globetrotter, passionate about creating unforgettable travel experiences!",
        verbose=True,
        allow_delegation=True,
        tools=[duckduckgo_search],
        LLM="gpt-3.5-turbo"
    )

    city_explorer = Agent(
        role="City Explorer",
        goal=f"Explore potential destinations and suggest exciting cities based on your interests.",
        backstory="An expert in uncovering hidden travel gems, ready to find the perfect city for your trip!",
        verbose=True,
        allow_delegation=True,
        tools=[duckduckgo_search],
        LLM="gpt-3.5-turbo"
    )

    activity_scout = Agent(
        role="Activity Scout",
        goal=f"Find exciting activities and attractions in {destination} that match your interests.",
        backstory=f"An expert curator of unique experiences, ready to unveil the hidden gems of {destination}.",
        verbose=True,
        allow_delegation=True,
        tools=[duckduckgo_search],
        LLM="gpt-3.5-turbo"
    )

    logistics_coordinator = Agent(
        role="Logistics Coordinator",
        goal=f"Help you navigate the logistics of your trip to {destination}, including flights, accommodation, and transportation.",
        backstory="A logistical whiz, ensuring your trip runs smoothly from start to finish.",
        verbose=True,
        allow_delegation=True,
        tools=[duckduckgo_search],
        LLM="gpt-3.5-turbo"
    )

    # Define Tasks
    if destination:
        task1 = Task(
            description=f"Plan a personalized itinerary for a trip to {destination}. Consider preferences like travel style (adventure, relaxation, etc.), budget, and desired activities.",
            expected_output=f"Personalized itinerary for a trip to {destination}.",
            agent=travel_advisor,
            LLM="gpt-3.5-turbo"
        )
    else:
        task1 = Task(
            description="Help you explore potential destinations and suggest exciting cities to visit based on your interests (e.g., beaches, culture, nightlife).",
            expected_output="List of recommended destinations and interesting cities to visit.",
            agent=city_explorer,
            LLM="gpt-3.5-turbo"
        )

    task2 = Task(
        description=f"Find exciting activities and attractions in {destination} that align with your preferences (e.g., museums, hiking, nightlife).",
        expected_output=f"List of recommended activities and attractions in {destination}.",
        agent=activity_scout,
        LLM="gpt-3.5-turbo"
    )

    task3 = Task(
        description=f"Help you navigate the logistics of your trip to {destination}. Search for flights, accommodation options, and local transportation based on your preferences and itinerary.",
        expected_output=f"Recommendations for flights, accommodation, and transportation for your trip to {destination}.",
        agent=logistics_coordinator,
        LLM="gpt-3.5-turbo"
    )

    # Create and Run the Crew
    travel_crew = Crew(
        agents=[travel_advisor, city_explorer, activity_scout, logistics_coordinator],
        tasks=[task1, task2, task3],
        verbose=True,
        process=Process.sequential,
        LLM="gpt-3.5-turbo"
    )

    crew_result = travel_crew.kickoff()
    return crew_result

if __name__ == "__main__":
    destination = "Paris"  # You can change this or make it a command-line argument
    logger.info(f"Creating travel crew for destination: {destination}")
    result = create_travel_crew(destination)
    logger.info("Travel Crew Result:")
    logger.info(result)