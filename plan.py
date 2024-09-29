import logging
import re
import sys

import streamlit as st
from crewai import LLM, Agent, Crew, Process, Task
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up the custom LLM
custom_llm = LLM(
    model="openai/phi",
    base_url="https://phi.us.gaianet.network/v1",
    api_key="any_value"
)

# Define a custom prompt template
custom_prompt = PromptTemplate.from_template("""
You are an AI assistant tasked with {role}. Your goal is to {goal}.

Here's some context about you: {backstory}

When responding, follow these rules:
1. Think step-by-step about the task at hand.
2. If you need to perform an action, use the format:
   Action: [action name]
   Action Input: [input for the action]
3. If you have a final answer or recommendation, use the format:
   Final Answer: [your answer or recommendation]
4. Never combine an Action and a Final Answer in the same response.

Now, please address the following task:
{task}

Begin your response:
""")


class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']
        self.color_index = 0

    def write(self, data):
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()
        if task_value:
            st.toast(":robot_face: " + task_value)
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            self.color_index = (self.color_index + 1) % len(self.colors)
            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")
        for role in ["Travel Advisor", "City Explorer", "Activity Scout", "Logistics Coordinator"]:
            if role in cleaned_data:
                cleaned_data = cleaned_data.replace(role, f":{self.colors[self.color_index]}[{role}]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")
        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

def create_agent(role, goal, backstory, llm):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[],
        agent_kwargs={
            "handle_parsing_errors": True,
            "prompt": custom_prompt
        },
        # callbacks=[StreamlitCallbackHandler(role)]
    )

def create_travel_crew(destination):
    # Define Agents
    travel_advisor = create_agent(
        role="Travel Advisor",
        goal=f"Craft a personalized itinerary for a trip to {destination}",
        backstory="A seasoned globetrotter, passionate about creating unforgettable travel experiences!",
        llm=custom_llm
    )

    city_explorer = create_agent(
        role="City Explorer",
        goal=f"Explore potential destinations and suggest exciting cities based on interests",
        backstory="An expert in uncovering hidden travel gems, ready to find the perfect city for your trip!",
        llm=custom_llm
    )

    activity_scout = create_agent(
        role="Activity Scout",
        goal=f"Find exciting activities and attractions in {destination} that match interests",
        backstory=f"An expert curator of unique experiences, ready to unveil the hidden gems of {destination}.",
        llm=custom_llm
    )

    logistics_coordinator = create_agent(
        role="Logistics Coordinator",
        goal=f"Help navigate the logistics of a trip to {destination}",
        backstory="A logistical whiz, ensuring your trip runs smoothly from start to finish.",
        llm=custom_llm
    )

    # Define Tasks
    if destination:
        task1 = Task(
            description=f"Plan a personalized itinerary for a trip to {destination}. Consider preferences like travel style (adventure, relaxation, etc.), budget, and desired activities.",
            expected_output=f"Personalized itinerary for a trip to {destination}.",
            agent=travel_advisor
        )
    else:
        task1 = Task(
            description="Help explore potential destinations and suggest exciting cities to visit based on interests (e.g., beaches, culture, nightlife).",
            expected_output="List of recommended destinations and interesting cities to visit.",
            agent=city_explorer
        )

    task2 = Task(
        description=f"Find exciting activities and attractions in {destination} that align with preferences (e.g., museums, hiking, nightlife).",
        expected_output=f"List of recommended activities and attractions in {destination}.",
        agent=activity_scout
    )

    task3 = Task(
        description=f"Help navigate the logistics of the trip to {destination}. Consider flights, accommodation options, and local transportation based on preferences and itinerary.",
        expected_output=f"Recommendations for flights, accommodation, and transportation for the trip to {destination}.",
        agent=logistics_coordinator
    )

    # Create and Run the Crew
    travel_crew = Crew(
        agents=[travel_advisor, city_explorer, activity_scout, logistics_coordinator],
        tasks=[task1, task2, task3],
        verbose=True,
        process=Process.sequential
    )

    crew_result = travel_crew.kickoff()
    return crew_result

# Streamlit UI
st.title("🌍 AI Travel Planner")


with st.sidebar:
    st.header("👇 Enter your trip details")
    with st.form("my_form"):
        destination = st.text_input("Enter your destination:", "Paris")
        submitted = st.form_submit_button("Plan My Trip")
        
if submitted:
    with st.status("🤖 **Planning your trip...**", state="running", expanded=True) as status:
        with st.container(height=500, border=False):
            expander = st.empty()
            # sys.stdout = StreamToExpander(expander)
            result = create_travel_crew(destination)
        status.update(label="✅ Trip Plan Ready!", state="complete", expanded=False)
    st.success("Your travel plan is ready!")
    st.subheader("Here is your Trip Plan", anchor=False, divider="rainbow")
    st.markdown(result)

if __name__ == "__main__":
    logger.info("Travel Planner app is running")