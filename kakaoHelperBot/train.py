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



LOADER_DICT = {
  "py": TextLoader,
  "md": UnstructuredMarkdownLoader,
  "ipynb": NotebookLoader,
  "txt": TextLoader
}

def upload_embedding_from_file(file_path):
  loader = LOADER_DICT.get(file_path.split(".")[-1])
  if loader is None:
    raise ValueError("Not supported file type")
  documents = loader(file_path).load()

  text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  docs = text_splitter.split_documents(documents)
  print(docs, end='\n\n\n')

  Chroma.from_documents(
    docs,
    OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
    persist_directory=CHROMA_PERSIST_DIR,
  )
  print('db success')

def upload_embeddings_from_dir(dir_path):
  failed_upload_files = []

  for root, dirs, files in os.walk(dir_path):
    print(files)
    for file in files:
      if file.endswith(".py") or file.endswith(".md") or file.endswith(".ipynb") or file.endswith(".txt"):
        file_path = os.path.join(root, file)

        try:
          upload_embedding_from_file(file_path)
          print("SUCCESS: ", file_path)
        except Exception as e:
          print("FAILED: ", file_path + f"by({e})")
          failed_upload_files.append(file_path)


print("uploading..")
upload_embeddings_from_dir("datas")
print("finishing..")
