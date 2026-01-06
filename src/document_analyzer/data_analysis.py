import os
import sys
from utils.model_loader import ModelLoader
#from logger import GLOBAL_LOGGER as log
from exception.custom_exception_legacy import DocumentPortalException
from models.models import *
from langchain_core.output_parsers import JsonOutputParser

from langchain_core.output_parsers import PydanticOutputParser
#from langchain_core.output_parsers import JsonOutputParser, OutputFixingParser

from prompt.prompt_library import PROMPT_REGISTRY # type: ignore
from mylogger.custom_logger import CustomLogger

class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm()
            
            
            # Prepare parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            #self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            
            self.prompt = PROMPT_REGISTRY["document_analysis"]
            
            self.log.info("DocumentAnalyzer initialized successfully")
            
            
        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)
        
        
    
    def analyze_document(self, document_text:str)-> dict:
        """
        Analyze a document's text and extract structured metadata & summary.
        """
        try:
            chain = self.prompt | self.llm | self.parser
            
            self.log.info("Meta-data analysis chain initialized")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            self.log.info("Metadata extraction successful", keys=list(response.keys()))
            
            return response

        except Exception as e:
            self.log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed",sys)
        
    