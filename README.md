
## Overview

The system allows users to ask natural-language questions about company documents and returns:

- A concise answer
- The source document
- Page number
- Chunk information

The system is deployed using Streamlit.

## Architecture

```text
Company PDFs
     ↓
Text Extraction
     ↓
Document Chunking
     ↓
Sentence Transformer Embeddings
     ↓
Chroma Vector Database
     ↓
Top-k Retrieval
     ↓
Cross-Encoder Reranking
     ↓
Structured Answer Extraction
     ↓
FLAN-T5-large
     ↓
Answer + Sources

## RAG Pipeline

### 1. Document Processing

The system extracts text from the provided PDF documents and divides the content into meaningful chunks.

The final dataset contains 47 chunks across the provided documents.

### 2. Embeddings

Documents and user queries are converted into vector representations using:

`sentence-transformers/all-MiniLM-L6-v2`

### 3. Vector Database

ChromaDB is used to store document embeddings and retrieve relevant chunks for a given question.

### 4. Reranking

Retrieved documents are reranked using a Cross-Encoder to improve the relevance of the selected context.

### 5. Structured Answer Extraction

For simple factual and table-based questions, deterministic extraction rules are used for values such as:

- File upload limits
- API rate limits
- Burst allowances
- Storage limits
- Uptime guarantees
- Annual billing discounts

This reduces unnecessary LLM generation.

### 6. Answer Generation

For questions that require natural-language generation, the system uses:

`google/flan-t5-large`

The model is instructed to answer using only the retrieved document context.

### 7. Source Attribution

Each answer includes the relevant:

- Document
- Page
- Chunk

This improves transparency and allows users to trace the answer back to the source.

## Streamlit Application

The application provides:

- Question input
- Example questions
- Answer display
- Source information
- Clear button
- Expandable source sections

## Example Questions

- How do I reset my password?
- What is the monthly price of the Standard plan?
- What is the maximum file size I can upload?
- When is two-factor authentication mandatory?
- What happens if a Standard account exceeds its pooled storage?

## Evaluation

The system was evaluated using 20 test questions covering:

- Direct factual questions
- Table-based questions
- Numeric questions
- Policy questions
- Paraphrased questions
- Unanswerable questions

The evaluation results are stored in:

`rag_evaluation_results.csv`

## Technologies

- Python
- PyPDF
- Sentence Transformers
- ChromaDB
- Cross-Encoder
- Hugging Face Transformers
- FLAN-T5
- Streamlit
- Pandas

## Project Structure

```text
Atman-Cloud-Document-QA/
├── app.py
├── README.md
├── requirements.txt
├── data/
├── chroma_db/
└── rag_evaluation_results.csv

## Running the Application

Install dependencies:

```bash
pip install -r requirements.txt

## Running the Application

Install dependencies:

`pip install -r requirements.txt`

Run Streamlit:

`streamlit run app.py`

The application will then be available through the Streamlit URL.

## Limitations

The system relies on semantic retrieval and reranking, so highly indirect or unusual question phrasing may sometimes retrieve less relevant context.

For unsupported questions, the system is designed to return:

`I couldn't find this information in the provided documents.`
