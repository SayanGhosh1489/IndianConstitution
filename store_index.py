from src.helper import text_split, data_reader, download_embedding
from langchain_pinecone import PineconeVectorStore
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()
PINE_CONE_API = os.environ.get('PINE_CONE_API')
index = os.environ.get('PINE_CONE_INDEX')


extracted_data = data_reader("data/")
text_chunk = text_split(extracted_data)
embedding = download_embedding()


def create_vectordb(PINE_CONE_API, index, text_chunk = text_chunk, embedding = embedding):
    load_dotenv()
    os.environ['PINECONE_API_KEY'] = PINE_CONE_API
    index = index
    docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunk], embedding=embedding, index_name = index)
    return docsearch


docsearch = create_vectordb(PINE_CONE_API =PINE_CONE_API, index=index)