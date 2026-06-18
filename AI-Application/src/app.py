import argparse
import uuid
from typing import List
from .config import settings
from .blob_store import BlobStore
from .document import DocumentChunk, chunk_text
from .foundry_client import FoundryClient
from .vector_store import VectorStore


class RAGApplication:
    def __init__(self):
        self.blob_store = BlobStore(settings.azure_connection_string, settings.azure_blob_container)
        self.foundry = FoundryClient()
        self.vector_store = VectorStore(settings.vector_store_path)

    def ingest(self, prefix: str = "") -> None:
        blob_names = self.blob_store.list_blobs(prefix)
        if not blob_names:
            print("No blobs found for prefix:", prefix)
            return

        for blob_name in blob_names:
            print(f"Processing blob: {blob_name}")
            raw_text = self.blob_store.load_text_from_blob(blob_name)
            chunks = chunk_text(raw_text)
            texts = [chunk for chunk in chunks if chunk.strip()]
            if not texts:
                print(f"Skipped empty document: {blob_name}")
                continue

            embeddings = self.foundry.create_embeddings(texts)
            for chunk_text_value, embedding in zip(texts, embeddings):
                document_chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    source=blob_name,
                    text=chunk_text_value,
                )
                self.vector_store.add(document_chunk, embedding)
            print(f"Added {len(texts)} chunks from {blob_name}")

    def query(self, user_query: str) -> None:
        print("Encoding query...")
        query_embedding = self.foundry.create_embeddings([user_query])[0]
        results = self.vector_store.similarity_search(query_embedding, top_k=settings.rag_top_k)
        if not results:
            print("No relevant documents found.")
            return

        context = "\n\n".join([f"Source: {item['source']}\n{item['text']}" for item in results])
        prompt = (
            "Use the retrieved document passages to answer the question below. "
            "If the answer is not contained in the passages, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
        )

        answer = self.foundry.generate(prompt)
        print("\n=== Answer ===")
        print(answer)
        print("\n=== Sources ===")
        for item in results:
            print(f"- {item['source']}")

    def list_blobs(self, prefix: str = "") -> None:
        blob_names = self.blob_store.list_blobs(prefix)
        if not blob_names:
            print("No blobs found")
            return
        print("Blobs:")
        for blob in blob_names:
            print(f"- {blob}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG prototype using Azure Blob and Microsoft Foundry")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest blobs and create embeddings")
    ingest_parser.add_argument("--blob-prefix", default=settings.azure_blob_prefix, help="Blob prefix to ingest")

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="User query text")

    list_parser = subparsers.add_parser("list-blobs", help="List blobs in the container")
    list_parser.add_argument("--blob-prefix", default=settings.azure_blob_prefix, help="Blob prefix to list")

    args = parser.parse_args()
    app = RAGApplication()

    if args.command == "ingest":
        app.ingest(prefix=args.blob_prefix)
    elif args.command == "query":
        app.query(args.query)
    elif args.command == "list-blobs":
        app.list_blobs(prefix=args.blob_prefix)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
