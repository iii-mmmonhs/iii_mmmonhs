import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

PDF_PATH = os.path.join(PROJECT_DIR, "data", "IBM_SPSS_Statistics_Core_System_User_Guide.pdf")
EMBEDDINGS_PATH = os.path.join(PROJECT_DIR, "embeddings", "index.faiss")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 512
OVERLAP = 32