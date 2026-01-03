import sys
from pathlib import Path
from mylogger.custom_logger import CustomLogger
from exception.custom_exception_legacy import DocumentPortalException
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
from datetime import datetime,timezone
import uuid
import shutil
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.model_loader import ModelLoader
class SingleDocIngestor:
    def __init__(self,data_dir: str = "data/single_doc_chat",faiss_dir: str ="faiss-index"):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.data_dir =Path(data_dir)
            self.data_dir.mkdir(parents=True,exist_ok=True)
            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True,exist_ok=True)
            self.model_loader = ModelLoader()
            self.log.info("Single doc Ingestor Initialized",temp_path=str(self.data_dir),faiss_path = str(self.faiss_dir))
        except Exception as e:
            self.log.error(f"Error intializing single document ingestor: {e}")
            raise DocumentPortalException("intializing error in single document ingestion",sys)
        
    def ingest(self,uploaded_files):
        try:
            documents =[]
            for uploaded_file in uploaded_files:
                unique_filename = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                temp_path = self.data_dir / unique_filename

                with open(temp_path,"wb") as f_out:
                    f_out.write(uploaded_file.read())
                self.log.info("pdf ingested successfully",filename = (uploaded_file).name)
                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()
                documents.extend(docs)
                self.log.info("pdf file loaded",count = len(documents))
                return self._create_retriever(documents)

        except Exception as e:
            self.log.error(f"Error intializing sigle doc ingestor: {e}")
            raise DocumentPortalException("intializing error in single document ingestior",sys)

    def _create_retriever(self,documents):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap=300)
            chunks = splitter.split_documents(documents)
            self.log.info("documents converted into chunks",count = len(chunks))
            loader=ModelLoader()
            self.embeddings = loader.load_embeddings()
            self.log.info("embedding model loaded",embmodel=self.embeddings)
            vector_store = FAISS.from_documents(documents=chunks,embedding=self.embeddings)
            vector_store.save_local(str(self.faiss_dir))
            retriver = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})
            self.log.info("retriever created succesfully",retriver_type = (str(type(retriver))))
            return retriver
        except Exception as e:
            self.log.error(f"Error intializing sigle doc retirver: {e}")
            raise DocumentPortalException("intializing error in single document retriever",sys)
