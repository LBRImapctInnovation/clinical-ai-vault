import os
import io
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Azure SDKs
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

# 1. Unlock the Vault
load_dotenv()

print("Initializing Vault Connections...")

# Connect to the Storage Filing Cabinet
blob_service = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
container_client = blob_service.get_container_client("clinical-protocols")

# Connect to the AI Mathematician (Embeddings)
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,
    api_version="2024-02-01",
)

# Connect to the AI Librarian (Search)
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
search_credential = AzureKeyCredential(search_key)
index_name = "clinical-index"


def build_librarian_filing_system():
    """Tells Azure Search how to organize our vectors"""
    print("Building Vector Search Index in Azure...")
    index_client = SearchIndexClient(
        endpoint=search_endpoint, credential=search_credential
    )

    # Define the columns for our database
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="sourcefile", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    # Configure the complex vector math algorithm
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile", algorithm_configuration_name="myHnsw"
            )
        ],
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print("Librarian Filing System Ready!")


def process_pdf(filename):
    """The main RAG Pipeline"""
    print(f"\n1. Securely pulling [{filename}] from the Azure Vault...")
    blob_client = container_client.get_blob_client(filename)
    pdf_bytes = blob_client.download_blob().readall()

    print("2. Cracking open the PDF and reading all pages...")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    print("3. Slicing document into readable clinical chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(full_text)

    print(
        f"   -> Sliced into {len(chunks)} paragraphs. Translating to mathematical vectors..."
    )

    documents = []
    for i, chunk in enumerate(chunks):
        # Have the Mathematician convert text to vectors
        response = openai_client.embeddings.create(
            input=chunk, model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        embedding = response.data[0].embedding

        # Package it for the Librarian
        documents.append(
            {
                "id": f"chunk-{i}",
                "content": chunk,
                "sourcefile": filename,
                "content_vector": embedding,
            }
        )

    print("4. Uploading Vectors to the AI Search Librarian...")
    search_client = SearchClient(
        endpoint=search_endpoint, index_name=index_name, credential=search_credential
    )
    search_client.upload_documents(documents)
    print("\n✅ SUCCESS! The AI Brain can now read this PDF!")


# --- EXECUTE THE PIPELINE ---
build_librarian_filing_system()
process_pdf("Alhassen et al. 2021, C. yanhusuo, pain.pdf")
