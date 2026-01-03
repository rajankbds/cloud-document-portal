import os
import sys
import json
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from exception.custom_exception_legacy import DocumentPortalException
from mylogger.custom_logger import CustomLogger

log=CustomLogger().get_logger(__name__)
dotenv_path="/Users/rajankumar/MM/azure.env"
load_dotenv(dotenv_path)
openai_api_key=os.getenv("OPENAI_API_KEY")
class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
       self.openai_api_key = openai_api_key
       self.config = load_config()
       self.log = CustomLogger().get_logger(__name__)

    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """
        try:
            self.model_name = self.config["embedding_model"]["model_name"]
            
            self.log.info("Loading embedding model", model=self.model_name)
            return OpenAIEmbeddings(model=self.model_name,
                                                openai_api_key=openai_api_key) #type: ignore
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        #print(llm_block)
        provider_key = os.getenv("LLM_PROVIDER", "openai")
        #print(provider_key)

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        #print(llm_config)
        provider = llm_config.get("provider")
       # print(provider)
        model_name = llm_config.get("model_name")
        #print(model_name)
        temperature = llm_config.get("temperature", 0.05)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM provider=%s model=%s", provider, model_name)


    

        if provider == "openai":
                           return ChatOpenAI(
                            model=model_name,
                 api_key=self.openai_api_key,
                 temperature=temperature,
                 max_tokens=max_tokens
             )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


#if __name__ == "__main__":
    #loader = ModelLoader()

   

    # Test LLM
    #llm = loader.load_llm()
    #print(f"LLM Loaded: {llm}")
    #result = llm.invoke("Hello, how are you?")
    #print(f"LLM Result: {result.content}")
             # Test Embedding
    #embeddings = loader.load_embeddings()
    #print(f"Embedding Model Loaded: {embeddings}")
    ##result = embeddings.embed_query("Hello, how are you?")
    #print(f"Embedding Result: {result}")
