import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    azure_connection_string: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    azure_blob_container: str = os.getenv("AZURE_BLOB_CONTAINER_NAME", "")
    azure_blob_prefix: str = os.getenv("AZURE_BLOB_PREFIX", "")
    foundry_api_key: str = os.getenv("FOUNDRY_API_KEY", "")
    foundry_embedding_url: str = os.getenv("FOUNDRY_EMBEDDING_URL", "")
    foundry_llm_url: str = os.getenv("FOUNDRY_LLM_URL", "")
    foundry_embedding_model: str = os.getenv("FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-small")
    foundry_llm_model: str = os.getenv("FOUNDRY_LLM_MODEL", "gpt-35-turbo")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "vector_store.json")

    @property
    def auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.foundry_api_key}",
            "Content-Type": "application/json",
        }

settings = Settings()
