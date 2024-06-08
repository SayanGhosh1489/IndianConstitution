from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def data_reader(Data):
    """
    docment loader function.
    package reeuire: from langchain.document_loaders import PyPDFLoader, DirectoryLoader
    """
    loader = DirectoryLoader(Data,
                    glob= '*.pdf',
                    loader_cls= PyPDFLoader)
    doc = loader.load()
    return doc

def text_split(extracted_data, size = 500, overlap = 100):
    """
    Splitting the data into text chunks
    default Chunk Size = 500, chunk_overlap = 50
    Package required: from langchain.text_splitter import RecursiveCharacterTextSplitter 
    Returns: text_chunks
    """
    text_spilter = RecursiveCharacterTextSplitter(chunk_size = size, chunk_overlap = overlap)
    text_chunks = text_spilter.split_documents(extracted_data)
    return text_chunks


def download_embedding():
    """
    Downloading embedding model from HuggingFace
    Package required: from langchain.embeddings import HuggingFaceBgeEmbeddings
    """

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding