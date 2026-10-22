import os
from dotenv import load_dotenv

load_dotenv()

# Renamed from OPENAI_API_KEY — value was a Groq key (gsk_...).
# Will be used for Groq/OpenAI routing in future phases.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

INDEX_NAME = "clinical-trial-index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384