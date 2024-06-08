from src.helper import text_split, data_reader, download_embedding
from langchain_pinecone import PineconeVectorStore
import pinecone
import os
from dotenv import load_dotenv

PINE_CONE_API = os.environ.get('PINE_CONE_API')
index = os.environ.get('PINE_CONE_INDEX')

extracted_data = load_pdf("data/")
text_chunk = text_split(extracted_data)
embedding = download_hugging_face_embeddings()


def create_vectordb(PINE_CONE_API, index, text_chunk = text_chunk, embedding = embedding):
    load_dotenv()
    os.environ['PINECONE_API_KEY'] = PINE_CONE_API
    index = index
    docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunk], embedding=embedding, index_name = index)
    return docsearch

