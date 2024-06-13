import chainlit as cl
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
from langchain.llms import CTransformers
import os
from langchain.chains import RetrievalQA
from src.prompt import prompt_creator
import warnings
warnings.filterwarnings('ignore')

embeddings = download_embedding()
prompt = prompt_creator()

#pine cone
load_dotenv()

PINE_CONE_API = os.environ.get('PINE_CONE_API')
index = os.environ.get('PINE_CONE_INDEX')

#Loading the index
os.environ['PINECONE_API_KEY'] = PINE_CONE_API

def get_model():
    llm = CTransformers(model="Model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':1024,
                          'temperature':0.8,
                          "top_p": 0.9
                          },device = 'auto')
    return llm


def qa_bot():
    docsearch=PineconeVectorStore.from_existing_index(index_name=index,embedding=embeddings)
    llm=get_model()
    qa=RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True, 
        chain_type_kwargs=prompt)
    return qa

def final_results(query):
    qa_result = qa_bot()
    response = qa_result({'query' : query})
    return response


#chainlit
@cl.on_chat_start
async def start():
    chain = qa_bot()  # Ensure qa_bot is defined and returns a valid chain object
    msg = cl.Message(content="Starting the bot....")
    await msg.send()
    msg.content = "Hi, Welcome to Indian Constitution. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)  # Correct the typo here

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # Retrieve the chain object correctly
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]  # Correct the parameter name to answer_prefix_tokens
    )

    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res['result']
    print(answer)
    sources = res['source_documents']
    
    if sources:
        answer += f"\n\nSources: " + str(sources)
    else:
        answer += "\nNo source found"
        
    await cl.Message(content=answer).send()