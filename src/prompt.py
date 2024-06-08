
from langchain import PromptTemplate
from langchain.prompts import PromptTemplate


def prompt_creator():
    """
    Creating Prompt for the llm
    Package Required: from langchain import PromptTemplate
    return: dict
    """

    prompt_template="""
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs={"prompt": PROMPT}
    return chain_type_kwargs