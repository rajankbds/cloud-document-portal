import os
import sys
import pandas as pd
from utils.model_loader import ModelLoader
#from logger import GLOBAL_LOGGER as log
from exception.custom_exception_legacy import DocumentPortalException
from models.models import *
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from models.models import SummaryResponse,PromptType
from prompt.prompt_library import PROMPT_REGISTRY # type: ignore
from mylogger.custom_logger import CustomLogger

class DocumentComparatorLLM:
    def __init__(self):
        #load_dotenv()
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.log = CustomLogger().get_logger(__name__)
        #self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.parser
        self.log.info("DocumentComparatorLLM initialized")

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }

            self.log.info("Invoking document comparison LLM chain")
            response = self.chain.invoke(inputs)
            self.log.info("Chain invoked successfully", response_preview=str(response)[:200])
            return self._format_response(response)
        except Exception as e:
            self.log.error("Error in compare_documents", error=str(e))
            raise DocumentPortalException("Error comparing documents", sys)
        
    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame: #type
        try:
            df = pd.DataFrame(response_parsed)
            return df
        except Exception as e:
            self.log.error("Error formatting response into DataFrame", error=str(e))
            DocumentPortalException("Error formatting response", sys)