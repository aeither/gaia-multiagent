import os

from crewai import LLM, Agent, Crew, Task
from crewai_tools import tool
from dotenv import load_dotenv
from openai import OpenAI
from upstash_vector import Index

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai = OpenAI(
    api_key="no_key",
    base_url="https://llama.us.gaianet.network/v1"
)


custom_llm = LLM(
    model="openai/phi",
    base_url="https://phi.us.gaianet.network/v1",
    api_key="any_value",
)

# Initialize Upstash Vector index
index = Index(
    url=os.getenv("UPSTASH_VECTOR_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)

def get_embedding(text, model="text-embedding-1024"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

@tool
def rag_tool(question: str) -> str:
    """Useful for answering questions based on retrieved context from a vector database."""
    question_embedding = get_embedding(question)
    results = index.query(vector=question_embedding, top_k=3, include_metadata=True)
    context = "\n".join([r.metadata['text'] for r in results])
    
    prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
    
    print(question)
    print(prompt)
    # response = openai.chat.completions.create(
    #     model="llama",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant. Answer the question based only on the provided context. If you can't find the answer in the context, say 'I don't have enough information to answer that question.'"},
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    
    # answer = response.choices[0].message.content
    return prompt

# Create an agent with the custom RAG tool
researcher = Agent(
    role="Researcher",
    goal="You answer questions based on retrieved information.",
    backstory="You're an expert researcher with access to a vast database of information.",
    tools=[rag_tool],
    allow_delegation=False,
    llm=custom_llm
)

# Create a task that uses the RAG tool
research_task = Task(
    description="Answer the question: Who is Debby in the album 'Waltz for Debby'?",
    expected_output="A detailed answer based on the retrieved information.",
    agent=researcher
)

# Create and run the crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task]
)

result = crew.kickoff()
print(result)