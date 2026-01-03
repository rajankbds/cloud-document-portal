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
    def __init__(self,session_id:str,retriver=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.retriver = retriver
            self.llm = self.load_llm()
            self.contextulaize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            # Lazy pieces
            self.retriver= retriver
            self.chain = None
            if self.retriver is not None:
                self._build_lcel_chain()
            self._build_lcel_chain()

        except Exception as e:
            self.log.error(f"Error intializing multi document  ingestor: {e}")
            raise DocumentPortalException("intializing error in multi document ingestion",sys)
        

    def built_retriver(self,index_path:str):
        try:
            loader=ModelLoader()
            self.embeddings = loader.load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError("FAISS Index Not found")
            vectorstore=FAISS.load_local(index_path,self.embeddings,allow_dangerous_deserialization=True)
            self.retriver = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})
            self.log.info("retriever created succesfully",index_path=index_path)
            #self._build_lcel_chain()
            return self.retriver
        except Exception as e:
            self.log.error(f"Error intializing multi document load retriver from faiss {e}")
            raise DocumentPortalException("intializing error in multi document load retriver from faiss",sys)
        
    def load_llm(self):
        try:
            loader = ModelLoader()
            self.llm = loader.load_llm()
            if not self.llm:
                raise ValueError("lLM could not be loaded")
            self.log.info("LLM loaded succesfully",session_id = self.session_id)
            return self.llm
        except Exception as e:
            self.log.error(f"Error intializing multi document load LLM {e}")
            raise DocumentPortalException("intializing error in multi document load LLM",sys)
        
    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
    
    def _build_lcel_chain(self):
        try:
            question_rewriter = (
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history")}
                    |self.contextulaize_prompt
                    |self.llm
                    |StrOutputParser()
            )
            retrived_docs =question_rewriter|self.retriver | self._format_docs
            self.chain = (
                { "context":retrived_docs,
                 "input":itemgetter("input"),
                 "chat_history":itemgetter("chat_history"),
                 }
                 | self.qa_prompt
                 | self.llm
                 | StrOutputParser()
                  )
            self.log.info("lcel chain initialized succesfully",session_id = self.session_id)
        except Exception as e:
            self.log.error(f"Error intializing multi document build_lcel_chain {e}")
            raise DocumentPortalException("intializing error in multi document build_lcel_chain LLM",sys)
        
    def invoke(self,user_input:str,chat_history:Optional[List[BaseMessage]]=None)-> str:
        try:
            chat_history= chat_history or []
            payload ={"input":user_input,"chat_history":chat_history}
            answer=self.chain.invoke(payload)
            if not answer:
                self.log.warning("No answer genrated")
                return "NO ANSWER"
            
            self.log.info("Answer succesfully generated from invoke",session_id = self.session_id,user_input=user_input)
            return answer
        except Exception as e:
            self.log.error(f"Error intializing multi document invoke {e}")
            raise DocumentPortalException("intializing error in multi document invoke",sys)
        