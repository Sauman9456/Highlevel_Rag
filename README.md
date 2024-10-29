# High-Level Support Solutions Retrieval API

This project provides a Retrieval-Augmented Generation (RAG) solution for efficiently querying support documents from [GoHighLevel's Help Center](https://help.gohighlevel.com/support/solutions). The solution includes data crawling, vector storage setup, and an advanced query-answering API built with FastAPI. The project is optimized to deliver relevant answers with citations, using techniques like vector similarity search, query augmentation, and content re-ranking.

## Project Overview

### 1. Web Crawling
The project starts with crawling the [GoHighLevel Help Center](https://help.gohighlevel.com/support/solutions) to extract support documents. This stage involves:

- **Script**: `crawler.py`
- **Process**:
  - Crawls all URLs starting from `https://help.gohighlevel.com/support/solutions`, using Selenium.
  - Saves all text content as Markdown files, excluding images and videos.
  - Stores output in the `content/crawl` directory with the following structure:
    - `processed_urls.json`: List of all processed URLs.
    - `scraped_content/`: Contains Markdown files for each URL.
    - `md_url_mapping.json`: Mapping between URLs and the corresponding Markdown files.

### 2. Vector Database Setup
The next stage is to process the Markdown files and store them in a vector database for efficient retrieval.

- **Database**: ChromaDB
- **Processing**:
  - Converts Markdown files into LangChain documents, with added metadata.
  - Metadata includes section summaries (generated using the GPT-4o-mini model) and document indexes (based on headers marked with `#`).
  - Splits documents at header levels (`#`) for finer granularity.
  - Stores the vector database in `content/crawl/high_level_support_solution_chroma_langchain_db`.

### 3. RAG Execution and API Setup
The final stage provides an API endpoint using FastAPI, which performs an advanced search with combined vector search, keyword search, and re-ranking.

- **Script**: `main.py`
- **Functionality**:
  - Takes a query string from the user and returns an answer with citations in JSON format: `{"answer": string, "citation": List of dict}`.
  - Uses GPT-4o-mini for domain-aware query augmentation, generating alternative queries based on document indexes.
  - Selects the top 10 documents from each of the four queries (original + 3 augmented).
  - Merges content from documents within the same URL, if applicable, and re-ranks them.
  - Passes the top 15 documents to GPT-4o-mini to generate the final response with answers and citations.

### Example API Response
The RAG system provides responses in the following format:

```json
{
  "answer": "To address A2P campaign rejection, follow these steps: ...",
  "citation": [
    {
      "title": "A2P 10DLC Campaign Approval Best Practices",
      "url": "https://help.gohighlevel.com/support/solutions/articles/48001229784"
    },
    {
      "title": "Fixing Failed Number Registrations (A2P Local and Toll-Free)",
      "url": "https://help.gohighlevel.com/support/solutions/articles/155000001454"
    }
  ]
}
```

## Getting Started

### Prerequisites
- Python 3.7+
- `requirements.txt` contains all necessary libraries (install with `pip install -r requirements.txt`).
- API keys for OpenAI and Cohere (add these keys to your environment).

### Running the API

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your_openai_key"
   export COHERE_API_KEY="your_cohere_key"
   ```

3. **Start the FastAPI Server**:
   ```bash
   uvicorn main:app --reload
   ```

4. **Access the API Documentation**:
   Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger documentation.

### Making API Requests

You can use `curl` to test the endpoint:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/get_answer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input_text": "a sub account needs to transfer their existing phone number to their new high level account they created on their own"
}'
```

## Testing and Evaluation

To evaluate the RAG model's performance, a Jupyter Notebook `ragas_testing.ipynb` is provided, which includes scripts to assess the quality of the responses. Refer to the **`eval_dict.json`** to get response against the given questions
