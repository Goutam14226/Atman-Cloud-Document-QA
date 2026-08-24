
# Atman Cloud Document Q&A

A Retrieval-Augmented Generation (RAG) based document question-answering system that allows users to ask natural-language questions about the provided Atman Cloud company documents and receive concise, source-grounded answers.

The system retrieves relevant information from the documents before generating an answer and provides source attribution including the document name, page number, and chunk ID.

The application is deployed using Streamlit.

---

## Overview

The goal of this project is to build a document question-answering system that can answer questions from a collection of company documents while minimizing unsupported or hallucinated answers.

The system processes the provided PDF documents, divides them into meaningful chunks, converts the chunks into vector embeddings, stores them in a ChromaDB vector database, retrieves relevant chunks for a user query, reranks the retrieved candidates using a Cross-Encoder, and then produces an answer using either deterministic extraction or a language model.

The system returns:

- A concise answer
- Source document
- Page number(s)
- Chunk ID

If the required information cannot be found in the provided documents, the system responds:

> I couldn't find this information in the provided documents.

---

## Architecture

```text
                         ┌──────────────────────┐
                         │   Company PDF Files  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Text Extraction    │
                         │        PyPDF         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Document Chunking    │
                         │ Section-aware / FAQ  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Sentence Transformer │
                         │ all-MiniLM-L6-v2     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      ChromaDB        │
                         │   Vector Database    │
                         └──────────┬───────────┘
                                    │
                                    │
                              User Question
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Query Embedding      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Top-k Semantic       │
                         │ Retrieval             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Cross-Encoder        │
                         │ Reranking            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌────────────────────────────┐
                         │ Answer Strategy Selection  │
                         └────────────┬───────────────┘
                                      │
                         ┌────────────┼────────────┐
                         │            │            │
                         ▼            ▼            ▼
                    FAQ Direct   Structured     FLAN-T5
                     Answer      Extraction      Generation
                         │            │            │
                         └────────────┼────────────┘
                                      │
                                      ▼
                         ┌──────────────────────┐
                         │ Answer + Sources     │
                         │ Document / Page /    │
                         │ Chunk                │
                         └──────────────────────┘
The system relies on semantic retrieval and reranking, so highly indirect or unusual question phrasing may sometimes retrieve less relevant context.

For unsupported questions, the system is designed to return:

`I couldn't find this information in the provided documents.`
