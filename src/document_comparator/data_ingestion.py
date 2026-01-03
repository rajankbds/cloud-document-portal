import sys
from pathlib import Path
from mylogger.custom_logger import CustomLogger
from exception.custom_exception_legacy import DocumentPortalException
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
from datetime import datetime
import uuid
import shutil


class DocumentIngestion():
    def __init__(self, base_dir: str = "data/document_compare",session_id=None):
        self.base_dir = Path(base_dir)
        self.log = CustomLogger().get_logger(__name__)
        #self.base_dir.mkdir(parents=True,exist_ok=True)
        self.session_id = session_id or (f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")
        #self.session_id = session_id or generate_session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.log.info("DocumentComparator initialized",session_path=str(self.session_path))
    def delete_existing_file(self):
        try:
            if self.base_dir.exists() and self.base_dir.is_dir():
                for file in self.base_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.log.info("file delted",path=str(file))
                self.log.info("directory cleaned",directory=str(self.base_dir))

        except Exception as e:
            self.log.error("Error in deleting existing file", error=str(e))

    def save_uploaded_files(self,refrence_file,actual_file):
        try:
            self.delete_existing_file()
            self.log.error("Deleted existing file")
            ref_path = self.session_path/refrence_file.name
            act_path =self.session_path/actual_file.name

            with open (ref_path,"wb") as f:
                f.write(refrence_file.getbuffer())
            with open (act_path,"wb") as f:
                f.write(actual_file.getbuffer())

            self.log.info("Files saved",refrence = str(ref_path),actual = str(act_path))
            return ref_path,act_path
            
        except Exception as e:
            self.log.error("Error in save uploaded files file", error=str(e))
    def read_pdf(self,pdf_path:Path)-> str:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            #if docs.is_encrypted:
                #raise ValueError(f"PDF is encrypted: {pdf_path.name}")
            page_texts = {}

            for doc in docs:
              page_num = doc.metadata.get("page")  # 0- or 1-based depending on loader version
              page_texts.setdefault(page_num, [])
              page_texts[page_num].append(doc.page_content)

              # If each doc is already a full page, you can just:
            page_texts = {doc.metadata["page"]: doc.page_content for doc in docs}
            self.log.info("PDF read successfully", file=str(pdf_path), pages=len(docs))
            return page_texts
        except Exception as e:
            self.log.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DocumentPortalException("Error reading PDF", e) from e
        
    def combine_documents(self) -> str:
        try:
            doc_parts = []
            content_dict ={}
            for filename in sorted(self.base_dir.iterdir()):
                if filename.is_file() and filename.suffix.lower() == ".pdf":
                    content_dict['filename'] = self.read_pdf(filename)
            for filename,content in content_dict.items():
                doc_parts.append(f"Document: {filename}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined", count=len(doc_parts))
            return combined_text
        except Exception as e:
            self.log.error("Error combining documents", error=str(e))
            raise DocumentPortalException("Error combining documents", e) from e
        
    def clean_old_sessions(self, keep_latest: int = 3):
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                self.log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            self.log.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("Error cleaning old sessions", e) from e
           
         
            
               
                
            
       

        
    