# DocQA on Bedrock

DocQA answers questions about your own text documents. It finds the passages most relevant to a question, hands them to a Claude model on Amazon Bedrock, and returns the answer together with the files it drew from.

The retrieval pipeline is written directly in Python, with no LangChain or LlamaIndex, so each step can be read and changed on its own: splitting text, creating embeddings, searching a vector store, building the prompt and calling the model. The application code is about 150 lines.

## Pipeline

```
Indexing (once)
  data/docs/*.txt -> split into chunks -> Titan embeddings -> Chroma

Answering (each question)
  question -> embedding -> 3 nearest chunks -> prompt -> Claude -> answer + source files
```

| Step | File | Details |
|---|---|---|
| Splitting | `app/chunker.py` | 500-character windows with 50 characters of overlap |
| Embedding and generation | `app/bedrock_client.py` | Titan Embed Text v2 and a Claude model through Bedrock |
| Retrieval and prompting | `app/rag.py` | Top 3 chunks from a persistent Chroma collection; the model is told to answer only from the context |
| Indexing | `app/ingest.py` | Reads every `.txt` file in `data/docs/` and stores embeddings |
| API | `app/main.py` | `GET /health` and `POST /chat` |

## Before you start

- Python 3.9 or newer
- An AWS account whose user or role may call `bedrock:InvokeModel`
- AWS credentials on your machine, for example from `aws configure`
- Model access turned on in the Bedrock console for your region (default `us-east-1`) for:
  - `amazon.titan-embed-text-v2:0`
  - a Claude model such as `anthropic.claude-3-5-sonnet-20241022-v2:0`

Model access has to be requested once per account and region before the API will respond.

## Installation

```bash
git clone https://github.com/afiaahmedAI/DocQA-Bedrock.git
cd DocQA-Bedrock

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # adjust region or model IDs if you need to
```

## Running it

Index the documents, then start the server:

```bash
python -m app.ingest
uvicorn app.main:app --reload
```

Ask a question:

```bash
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How long do customers have to return an item?"}'
```

The response has an `answer` and a `sources` list. Run `python -m app.ingest` again whenever the files in `data/docs/` change.

## Your own documents

Put `.txt` files in `data/docs/` and re-run the ingest step. The two sample files describe a made-up company and are only there so the project works out of the box.

## Layout

```
app/
  bedrock_client.py
  chunker.py
  ingest.py
  main.py
  rag.py
data/docs/        sample documents
chroma_db/        vector store (created when you ingest, not committed)
```

## Limitations

- Chunks are cut by character count and can split a sentence in two.
- Plain text only. There is no PDF or Word support.
- No re-ranking and no memory of earlier questions.
- Good next steps: sentence-aware splitting, PDF ingestion, hybrid keyword and vector search, and a small question set to measure answer quality.
