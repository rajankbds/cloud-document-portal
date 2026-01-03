import sys
from pathlib import Path
from datetime import datetime,timezone
from mylogger.custom_logger import CustomLogger
from exception.custom_exception_legacy import DocumentPortalException
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.model_loader import ModelLoader
import uuid

class DocumentIngestion:
    SUGGESTED_FILE_TYPE ={".pdf",".docx",".txt"}
    def __init__(self,data_dir: str = "data/multi_doc_chat",faiss_dir: str ="faiss-index",session_id: str | None = None):
        try:
         self.log = CustomLogger().get_logger(__name__)
         self.temp_dir =Path(data_dir)
         self.temp_dir.mkdir(parents=True,exist_ok=True)
         self.faiss_dir = Path(faiss_dir)
         self.faiss_dir.mkdir(parents=True,exist_ok=True)
         #session id 
         self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
         self.session_temp_dir = self.temp_dir/ self.session_id
         self.session_faiss_dir = self.faiss_dir / self.session_id
         self.session_temp_dir.mkdir(parents=True,exist_ok=True)
         self.session_faiss_dir.mkdir(parents=True,exist_ok=True)

         self.model_loader = ModelLoader()
         self.log.info("Document Ingestor initialized",temp_base = str(self.temp_dir),faiss_base= str(self.faiss_dir),session_id = self.session_id,temp_path=str(self.session_temp_dir))
                       
        except Exception as e:
            self.log.error(f"Error intializing multi document  ingestor: {e}")
            raise DocumentPortalException("intializing error in multi document ingestion",sys)


    def ingest_files(self,uploaded_files):
     try:
        documents =[]
        for uploaded_file in uploaded_files:
           ext = Path(uploaded_file.name).suffix.lower()
           if ext not in self.SUGGESTED_FILE_TYPE:
              self.log.warning("unsupported file type",filename=uploaded_file.name)
           unique_filename = f"{uuid.uuid4().hex[:8]}{ext}"
           temp_path = self.session_temp_dir / unique_filename

           with open(temp_path,"wb") as f:
              f.write(uploaded_file.read())
           self.log.info("File Ingested succesfully", filename=uploaded_file.name,saved_as = str(temp_path))
           if ext == ".pdf":
              loader = PyPDFLoader(str(temp_path))
           elif ext ==".docx":
              loader = Docx2txtLoader(str(temp_path))
           elif ext == ".txt":
              loader = TextLoader(str(temp_path),encoding="utf-8")
           else:
              self.log.warning ("unsupported file types",filename = uploaded_file.name)
              continue
           docs = loader.load()
           documents.extend(docs)
           if not documents:
              raise DocumentPortalException("No valid documents loaded",sys)
           return self._create_retriver(documents)
     except Exception as e:
            self.log.error(f"Error intializing ingest file for multidoc: {e}")
            raise DocumentPortalException("intializing error in multi  document retriever",sys)
        
           

    def _create_retriver(self,documents):
       try:
            splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap=300)
            chunks = splitter.split_documents(documents)
            self.log.info("documents converted into chunks",count = len(chunks))
            loader=ModelLoader()
            self.embeddings = loader.load_embeddings()
            self.log.info("embedding model loaded",embmodel=self.embeddings)
            vector_store = FAISS.from_documents(documents=chunks,embedding=self.embeddings)
            vector_store.save_local(str(self.session_faiss_dir))
            self.log.info("vector store saved locally",path=str(self.session_faiss_dir),session_id=self.session_id)
            retriver = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})
            self.log.info("retriever created succesfully",retriver_type = (str(type(retriver))))
            return retriver
       except Exception as e:
            self.log.error(f"Error intializing create_retriver file for multidoc: {e}")
            raise DocumentPortalException("intializing error in multi  document _create_retriever",sys)
       




       



        

