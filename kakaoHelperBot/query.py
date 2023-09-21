from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.document_loaders import (
  NotebookLoader,
  TextLoader,
  UnstructuredMarkdownLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

CHROMA_PERSIST_DIR = "hari-chroma"
CHROMA_COLLECTION_NAME = "hari-bot"

os.environ["OPENAI_API_KEY"] = open("../appkey.txt", "r").read()

_db = Chroma(
  persist_directory=CHROMA_PERSIST_DIR,
  embedding_function=OpenAIEmbeddings(), # openAiEmbeddings에 api key가 왜 필요하지?
  collection_name=CHROMA_COLLECTION_NAME,
)
_retriever = _db.as_retriever()

def query_db(query: str, use_retriever: bool = False) -> list[str]:
  if use_retriever:
    docs = _retriever.get_relevant_documents(query)
  else:
    docs = _db.similarity_search(query)

  str_docs = [doc.page_content for doc in docs]
  return str_docs


print(query_db("카카오소설은 어떤 기능이 있어?"))
