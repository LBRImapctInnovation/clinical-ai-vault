# 🛡️ LBR Clinical AI Vault

**Secure, Hallucination-Free Retrieval-Augmented Generation (RAG) for Medical Documents.**

This repository contains the source code for an Enterprise-grade, Zero-Trust Clinical Intelligence platform. It allows medical professionals and researchers to securely upload highly sensitive clinical trial protocols and medical journals, and query them using an advanced, verifiable AI architecture.

## 🚀 Enterprise Features

* **🔒 Zero-Trust Security:** Built on Microsoft Entra ID. No hardcoded API keys are used. All authentication runs through highly secure Service Principals. Data never touches the public internet.
* **🧠 Hallucination-Free RAG:** The GPT-4 model is strictly "leashed." If an answer cannot be mathematically located within the provided clinical documents, the system is hard-coded to return `"Data not found."`
* **📑 Verifiable Audit Trail:** Features a custom "Document Cracker" that parses PDFs page-by-page. Every AI response includes a strict citation (e.g., `[Page 7, Paragraph 5]`) and displays the exact source text in a side-by-side Proof Window.
* **📐 Advanced Vector Mathematics:** Utilizes `text-embedding-3-small` to translate English medical terminology into 1,536-dimensional vectors, hosted on Azure AI Search for high-speed, contextual clinical retrieval.

## 🏗️ Architecture & Tech Stack

* **Frontend:** Streamlit (Python)
* **LLM Engine:** Azure OpenAI (GPT-4o / GPT-4)
* **Embedding Model:** Azure OpenAI (`text-embedding-3-small`)
* **Vector Database:** Azure AI Search (Basic Tier, HNSW Algorithm)
* **Document Vault:** Azure Blob Storage (Private Container)
* **Identity & Access:** Microsoft Entra ID (DefaultAzureCredential)
* **Data Processing:** LangChain (Text Splitters), PyPDF

## 🛠️ Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/clinical-ai-vault.git](https://github.com/yourusername/clinical-ai-vault.git)
cd clinical-ai-vault