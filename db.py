import os

from dotenv import load_dotenv
from upstash_vector import Index

# Load environment variables
load_dotenv()

# Initialize the index
index = Index(
    url=os.getenv("UPSTASH_VECTOR_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)

def upsert_vector():
    index.upsert(
        vectors=[
            ("id1", "Enter data as string", {"metadata_field": "metadata_value"}),
        ]
    )
    print("Vector upserted successfully")

def query_vector():
    result = index.query(
        data="Enter data as string",
        top_k=1,
        include_vectors=True,
        include_metadata=True
    )
    print("Query result:", result)

if __name__ == "__main__":
    # Uncomment the line below to upsert a vector
    # upsert_vector()
    
    query_vector()