import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any
import streamlit as st
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception_legacy import DocumentPortalException
from mylogger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from models.models import PromptType
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_retrieval_chain

class ConversationalRAG:
    def __init__(self,session_id,retriver):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.retriver = retriver
            self.llm = self._load_llm()
            self.contextulaize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            self.history_aware_retriver = create_history_aware_retriever(self.llm,self.retriver,self.contextulaize_prompt)
            self.qa_chain = create_stuff_documents_chain(self.llm,self.qa_prompt)
            self.rag_chain = create_retrieval_chain(self.history_aware_retriver,self.qa_chain)
            self.log.info("rag_chain intialized",session_id=self.session_id)

            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            self.log.info("created runnable with message history",session_id=self.session_id)
        except Exception as e:
            self.log.error(f"Error intializing single document conversational RAG: {e}")
            raise DocumentPortalException("intializing error in single document conversationalRAG",sys)
        
    def _load_llm(self):
        try:
            loader = ModelLoader()
            self.llm = loader.load_llm()
            self.log.info("LLM loaded successfully",class_name = self.llm.__class__.__name__)
            return self.llm
        except Exception as e:
            self.log.error(f"Error intializing single document load llm: {e}")
            raise DocumentPortalException("intializing error in single document load llm",sys)
        
    def _get_session_history(self, session_id: str)-> BaseChatMessageHistory:
        try:
            if "store" not in st.session_state:
                st.session_state.store = {}
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
                self.log.info("New chat session history created",session_id=session_id)

            return st.session_state.store[session_id]


        except Exception as e:
            self.log.error(f"Error intializing single document session history: {e}")
            raise DocumentPortalException("intializing error in single document session history",sys)
        
    def load_retirver_from_faiss(self,index_path: str):
        try:
            self.embeddings = ModelLoader.load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"faiss directory not found {index_path}")
            vectorstore = FAISS.load_local(index_path,self.embeddings)
            return vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})
        except Exception as e:
            self.log.error(f"Error intializing single document retirver from faiss: {e}")
            raise DocumentPortalException("intializing error in single document retirver from faiss",sys)
    
    def invoke(self,user_input:str)->str:
        try:
            response=self.chain.invoke(
                {"input":user_input},
                config={"configurable":{"session_id": self.session_id}})
            answer = response.get("answer","No answer")
            if not answer:
                self.log.error("no answer received",session_id = self.session_id)
            if answer:
                self.log.info("Answer received ",session_id = self.session_id,user_input=user_input)
            return answer
            
        except Exception as e:
            self.log.error(f"Error intializing single document retirver from invoke: {e}")
            raise DocumentPortalException("intializing error in single document retirver from invoke",sys)
