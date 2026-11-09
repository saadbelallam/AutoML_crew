import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

def get_llm(temperature=None):
    kwargs = {} if temperature is None else {"temperature": temperature}
    return init_chat_model(os.environ["LLM_MODEL"], **kwargs)
    