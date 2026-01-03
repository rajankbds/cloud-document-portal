import os
import uuid
from datetime import datetime
from mylogger.custom_logger import CustomLogger
from exception.custom_exception_legacy import DocumentPortalException
from pathlib import Path
from io import BytesIO
import structlog
from langchain_community.document_loaders import PyPDFLoader


class DocHandler():
    """
    class for stooring and reading file
    """
    def __init__(self,data_dir=None,session_id=None):
        try:
         self.log = CustomLogger().get_logger(__name__)
         self.data_dir = data_dir or os.getenv(
            "DATA_STORAGE_PATH",
            os.path.join(os.getcwd(), "data", "document_analysis")
         )
         self.session_id = session_id or ("session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]f}")

         self.session_path = os.path.join(self.data_dir, self.session_id)
         os.makedirs(self.session_path, exist_ok=True)
         # Log initialization
         self.log.info(
            "PDFHandler initialized",session_id=self.session_id,session_path = self.session_path)
        except Exception as e:
           self.log.error(f"Error intializing Document Handler{e}")
    
    def save_pdf(self,uploaded_file):
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                
                    f.write(uploaded_file.getbuffer())
                
            self.log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            self.log.error("Failed to save PDF", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save PDF: {str(e)}", e) from e
        
    
    def read_pdf(self, pdf_path: str) -> str:
        try:
            
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
           
            full_text = "\n\n".join(doc.page_content for doc in docs)
            self.log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(docs))
            return full_text
        except Exception as e:
            self.log.error("Failed to read PDF", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process PDF: {pdf_path}", e) from e


        



