from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings #sentence transformers will work too
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstrores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Promt Template for QA retrieval for each vector stores

    """
    prompt = PromptTemplate.from_template("Say {foo}")
    prompt.format(foo="bar")
    prompt = PromptTemplate(input_variables=["context", "question"],template=custom_prompt_template)
    # prompt = prompt.format(adjective="funny", content="chickens")

    return prompt

def load_llm():
    llm= CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", temperature= 0.5, max_new_tokens =512, model_type ='llama'        
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain =  RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',  model_kwargs = {'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})
    return response

###### Chain lit ################
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Howdy !!!")
    await msg.send()
    msg.content = "Hi Welcome"
    await msg.update()
    cl.user_session.set("chain", chain)
@cl.on_message
async def main(message):
    chain = cl.user_session.set("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens = ['FINAL', 'ANSWER']        
    )
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    # sources = res["source_document"]
    # if 
    await cl.Message(content=answer).send()





    