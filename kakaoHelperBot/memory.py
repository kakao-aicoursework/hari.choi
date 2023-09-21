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


###############################################3
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

def load_conversation_history(conversation_id: str):
  file_path = f"history/{conversation_id}.json"
  return FileChatMessageHistory(file_path)


def log_user_message(history: FileChatMessageHistory, user_message: str):
  history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
  history.add_ai_message(bot_message)


def get_chat_history(conversation_id: str):
  history = load_conversation_history(conversation_id)
  memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="user_message",
    chat_memory=history,
  )

  return memory.buffer

def read_prompt_template(file_path: str) -> str:
  with open(file_path, "r") as f:
    prompt_template = f.read()

  return prompt_template
def create_chain(llm, template_path, output_key):
  return LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template(
      template=read_prompt_template(template_path)
    ),
    output_key=output_key,
    verbose=True,
  )

llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")
DEFAULT_RESPONSE_PROMPT_TEMPLATE="template/default.txt"

default_chain = create_chain(
  llm=llm, template_path=DEFAULT_RESPONSE_PROMPT_TEMPLATE, output_key="output"
)
def generate_answer(user_message, conversation_id: str='fa1010') -> dict[str, str]:
  history_file = load_conversation_history(conversation_id)

  context = dict(user_message=user_message)
  context["input"] = context["user_message"]
  context["related_documents"] = query_db(user_message)
  context["chat_history"] = get_chat_history(conversation_id)

  answer = default_chain.run(context)

  log_user_message(history_file, user_message)
  log_bot_message(history_file, answer)
  return {"answer": answer}

# print(query_db("카카오소설은 어떤 기능이 있어?"))

# print(generate_answer("너의 이름은?"))

import sys

userInput = ' '.join(sys.argv[1:])
print(generate_answer(userInput))

# python3 memory.py "기능 말해봐"