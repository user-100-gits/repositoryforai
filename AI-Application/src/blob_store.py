import io
import os
from typing import List
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader


class BlobStore:
    def __init__(self, connection_string: str, container_name: str):
        if not connection_string or not container_name:
            raise ValueError("Azure connection string and container name are required")
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service.get_container_client(container_name)

    def list_blobs(self, prefix: str = "") -> List[str]:
        return [blob.name for blob in self.container_client.list_blobs(name_starts_with=prefix)]

    def download_blob(self, blob_name: str) -> bytes:
        blob_client = self.container_client.get_blob_client(blob_name)
        stream = blob_client.download_blob()
        return stream.readall()

    def load_text_from_blob(self, blob_name: str) -> str:
        raw_bytes = self.download_blob(blob_name)
        extension = os.path.splitext(blob_name)[-1].lower()
        if extension in {".txt", ".md", ".csv", ".json"}:
            return raw_bytes.decode("utf-8", errors="ignore")
        if extension == ".pdf":
            return self._text_from_pdf(raw_bytes)
        raise ValueError(f"Unsupported file type for blob '{blob_name}': {extension}")

    def _text_from_pdf(self, raw_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
