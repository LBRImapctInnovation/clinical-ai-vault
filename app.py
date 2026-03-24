import streamlit as st
import os
import io
import re
from dotenv import load_dotenv

# Document Cracking Tools
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Azure SDKs
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# 1. Load the Vault
load_dotenv()

# 2. Connect the Azure Tools
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

ai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,
    api_version="2024-02-01",
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="clinical-index",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
)

# 3. Configure UI
st.set_page_config(page_title="LBR Clinical Vault", page_icon="🛡️", layout="wide")

# --- PANEL 1: LEFT SIDEBAR ---
with st.sidebar:
    try:
        st.image("LBR LOGO.jpg", use_container_width=True)
    except:
        st.warning("Logo missing")

    st.divider()
    st.subheader("📁 Data Ingestion")
    uploaded_file = st.file_uploader("Upload Trial Protocol (PDF)", type=["pdf"])

    if uploaded_file and st.button("Securely Upload & Analyze"):
        with st.spinner(f"Encrypting, reading, and indexing {uploaded_file.name}..."):
            try:
                # A. Save to Storage Vault
                blob_service = BlobServiceClient.from_connection_string(
                    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                )
                container = blob_service.get_container_client("clinical-protocols")
                blob = container.get_blob_client(uploaded_file.name)
                pdf_bytes = uploaded_file.getvalue()
                blob.upload_blob(pdf_bytes, overwrite=True)

                # B. Crack the PDF Page-by-Page
                reader = PdfReader(io.BytesIO(pdf_bytes))
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100
                )

                documents = []
                chunk_id_counter = 0
                safe_filename = re.sub(r"[^a-zA-Z0-9]", "-", uploaded_file.name)

                # C. Stamp every chunk with its exact Page AND Paragraph Number
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if not page_text:
                        continue

                    chunks = splitter.split_text(page_text)
                    # FIX 1: Count the paragraphs!
                    for para_num, chunk in enumerate(chunks, start=1):
                        enhanced_chunk = (
                            f"[Page {page_num}, Paragraph {para_num}] {chunk}"
                        )

                        embed_response = ai_client.embeddings.create(
                            input=enhanced_chunk,
                            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                        )

                        documents.append(
                            {
                                "id": f"{safe_filename}-{chunk_id_counter}",
                                "content": enhanced_chunk,
                                "sourcefile": uploaded_file.name,
                                "content_vector": embed_response.data[0].embedding,
                            }
                        )
                        chunk_id_counter += 1

                # D. Upload to the Librarian
                search_client.upload_documents(documents)
                st.success(
                    f"✅ {uploaded_file.name} indexed with precise paragraph tracking!"
                )

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

    st.divider()
    st.subheader("🔍 Active Documents")

    try:
        blob_service = BlobServiceClient.from_connection_string(
            os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
        container = blob_service.get_container_client("clinical-protocols")
        blob_list = container.list_blobs()
        has_docs = False
        for b in blob_list:
            st.checkbox(b.name, value=True)
            has_docs = True
        if not has_docs:
            st.write("Vault is currently empty.")
    except:
        st.write("Unable to connect to vault storage.")

# Define Columns
chat_col, proof_col = st.columns([7, 3])

if "source_proof" not in st.session_state:
    st.session_state.source_proof = (
        "No active query. Submit a question to view source documents."
    )

# --- PANEL 2: CENTER CHAT ---
with chat_col:
    st.title("Clinical Query Interface")
    st.info("Zero-Trust RAG Session Active. Connected to Azure Vector Search.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello Dr. Brown. My memory is now upgraded with exact Page and Paragraph tracking. What can I answer for you?",
            }
        ]

    message_container = st.container(height=600)

    with message_container:
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    if prompt := st.chat_input("Ask a clinical question..."):
        with message_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                try:
                    embed_response = ai_client.embeddings.create(
                        input=prompt,
                        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                    )
                    question_vector = embed_response.data[0].embedding

                    # FIX 2: Added top=3 to put a hard leash on the Librarian!
                    search_results = search_client.search(
                        search_text=prompt,
                        vector_queries=[
                            {
                                "vector": question_vector,
                                "k_nearest_neighbors": 3,
                                "fields": "content_vector",
                                "kind": "vector",
                            }
                        ],
                        select=["content", "sourcefile"],
                        top=3,
                    )

                    context_text = ""
                    proof_display = ""
                    for i, result in enumerate(search_results):
                        context_text += f"\n\nSource: {result['sourcefile']}\nExcerpt: {result['content']}"
                        proof_display += f"**Result {i+1} from {result['sourcefile']}**\n\n_{result['content']}_\n\n---\n\n"

                    st.session_state.source_proof = (
                        proof_display
                        if proof_display
                        else "No documents matched this query."
                    )

                    # FIX 3: Strict instructions for the AI to output the paragraph number
                    system_prompt = f"""You are an elite PharmD Architect. Answer the user's question strictly using ONLY the clinical document excerpts provided below. 
                    IMPORTANT RULES:
                    1. You MUST cite the specific document name AND the precise [Page X, Paragraph Y] format directly in your answer.
                    2. If the answer is not in the excerpts, say 'Data not found.' 
                    
                    CLINICAL EXCERPTS:
                    {context_text}"""

                    gpt_messages = [{"role": "system", "content": system_prompt}]
                    gpt_messages.extend(st.session_state.messages[-3:])

                    response = ai_client.chat.completions.create(
                        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                        messages=gpt_messages,
                        temperature=0.1,
                    )

                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(f"Vault Connection Error: {str(e)}")

# --- PANEL 3: RIGHT PROOF WINDOW ---
with proof_col:
    st.subheader("📑 Source Proof")
    st.caption("Verifiable audit trail to eliminate hallucinations.")
    with st.container(height=600):
        st.info(st.session_state.source_proof)
