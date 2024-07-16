from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

loader = DirectoryLoader("./transcripts", glob="**/*.txt")
documents = loader.load()

# Create embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    add_start_index=True,
)

# Split documents into chunks
texts = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding= embeddings,
    persist_directory="./db-fin-transcript")

print("vectorstore creation completed")
