# AI-Application

This repository contains a simple RAG prototype for Azure Blob Storage + Microsoft Foundry.

## Overview

- Ingests text data from Azure Blob Storage
- Creates embeddings using a Foundry embedding API
- Stores vectors in a simple local vector store
- Performs retrieval and sends context to a Foundry LLM generation API

## Setup

1. Create a Python virtual environment
2. Install dependencies:

```bash
cd AI-Application
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your credentials

```bash
cp .env.example .env
```

## Usage

### Ingest documents from Azure Blob Storage

```bash
python -m src.app ingest --blob-prefix ""
```

### Ask a question

```bash
python -m src.app query "What is the main purpose of the data?"
```

### List blobs

```bash
python -m src.app list-blobs --blob-prefix ""
```

## Notes

- This prototype is optimized for testing and demonstration
- For production, you should add proper auth, error handling, encryption, and indexing
