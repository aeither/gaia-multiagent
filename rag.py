import os

import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from upstash_vector import Index, Vector

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai = OpenAI(
    api_key="no_key",
    base_url="https://llama.us.gaianet.network/v1" 
)

# Initialize Upstash Vector index
index = Index(
    url=os.getenv("UPSTASH_VECTOR_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)

def token_len(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_embedding(text, model="text-embedding-1024"):  # Use a model that produces 1024-dim vectors
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def get_embeddings(chunks, model="text-embedding-1024"):  # Use a model that produces 1024-dim vectors
    chunks = [c.replace("\n", " ") for c in chunks]
    response = openai.embeddings.create(input=chunks, model=model)
    return [r.embedding for r in response.data]

def create_mock_text():
    return """
    Bill Evans was an American jazz pianist and composer who is widely considered to be one of the most influential figures in the history of jazz piano. Born in 1929 in Plainfield, New Jersey, Evans began playing piano at a young age and quickly showed a natural talent for the instrument.

    Evans studied classical piano and composition at Southeastern Louisiana University, where he graduated in 1950. After a brief stint in the Army, he moved to New York City to pursue a career in jazz. In the late 1950s, Evans joined the Miles Davis Sextet, where he played on the landmark album "Kind of Blue."

    One of Evans' most famous albums is "Waltz for Debby," recorded live at the Village Vanguard in New York City in 1961. The album features Evans' trio with bassist Scott LaFaro and drummer Paul Motian. The title track, "Waltz for Debby," was written by Evans for his niece Debby.

    Throughout his career, Evans developed a unique and influential style of jazz piano playing characterized by his use of impressionistic harmony, introspective melodies, and a light, "singing" touch. He was known for his ability to create complex harmonies and his use of block chords, which became a hallmark of his style.

    Evans struggled with drug addiction for much of his life, which affected his health and career. Despite this, he continued to perform and record until his death in 1980 at the age of 51. His legacy continues to influence jazz pianists and musicians to this day.
    """

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        length_function=token_len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def upsert_vectors(chunks):
    vectors = []
    embeddings = get_embeddings(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector = Vector(id=f"chunk-{i}", vector=embedding, metadata={"text": chunk})
        vectors.append(vector)
    index.upsert(vectors)
    print(f"{len(vectors)} vectors upserted successfully")

def ask_question(question):
    question_embedding = get_embedding(question)
    results = index.query(vector=question_embedding, top_k=3, include_metadata=True)
    context = "\n".join([r.metadata['text'] for r in results])
    
    prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
    
    response = openai.chat.completions.create(
        model="llama",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question based only on the provided context. If you can't find the answer in the context, say 'I don't have enough information to answer that question.'"},
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = response.choices[0].message.content
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    # Create mock text
    mock_text = create_mock_text()
    
    # Chunk text
    chunks = chunk_text(mock_text)
    
    # Generate embeddings and upsert vectors
    upsert_vectors(chunks)
    
    # Ask questions
    ask_question("Who is Debby in the album 'Waltz for Debby'?")
    ask_question("Where did Bill Evans study?")