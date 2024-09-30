from crewai import LLM, Agent, Crew, Task
from crewai_tools import tool

custom_llm = LLM(
    model="openai/phi",
    base_url="https://phi.us.gaianet.network/v1",
    api_key="any_value",
)

@tool
def multiplication_tool(first_number: int, second_number: int) -> str:
    """Useful for when you need to multiply two numbers together."""
    return str(first_number * second_number)

# Create an agent with the custom tool
writer1 = Agent(
    role="Writer",
    goal="You write lessons of math for kids.",
    backstory="You're an expert in writing and you love to teach kids but you know nothing of math.",
    tools=[multiplication_tool],
    allow_delegation=False,
    llm=custom_llm
)

# Create a task that uses the tool
calculation_task = Task(
    description="Calculate the result of (23 * 4) + 7",
    agent=writer1,
    expected_output="The result of the calculation as a number."
)

# Create and run the crew
crew = Crew(
    agents=[writer1],
    tasks=[calculation_task]
)

result = crew.kickoff()
print(result)