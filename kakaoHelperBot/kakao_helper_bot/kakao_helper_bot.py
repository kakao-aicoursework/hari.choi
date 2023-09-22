"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import openai
import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

from langchain.utilities import DuckDuckGoSearchAPIWrapper
import tiktoken

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

CHROMA_PERSIST_DIR = "hari-chroma"
CHROMA_COLLECTION_NAME = "hari-bot"

openai.api_key = open("../appkey.txt", "r").read()
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
llm = ChatOpenAI(temperature=0.8)
DEFAULT_RESPONSE_PROMPT_TEMPLATE="template/default.txt"
info = open("datas/project_data_카카오싱크.txt", "r").read()

default_chain = create_chain(
    llm=llm, template_path=DEFAULT_RESPONSE_PROMPT_TEMPLATE, output_key="output"
)

functions = [{
    "name": "kakao_sink",
    "description": "kakao 의 신규 서비스 카카오싱크에 대한 전반적인 정보를 가져옵니다",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
},
    {
    "name": "kakao_social",
    "description": "kakao 의 신규 서비스 카카오소셜과 관련된 전반적인 정보를 가져옵니다",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
]

def generate_answer(user_message, conversation_id: str='fa1010') -> dict[str, str]:
    intend_msg = [{"role": "user", "content": user_message}] # 히스토리 넣자!

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=intend_msg,
        functions=functions,
        function_call="auto",
        max_tokens=4096
    )

    target = ""
    if "function_call" in completion['choices'][0]['message']:
        function_name = completion["choices"][0]["message"]["function_call"]["name"]
        print(f"call {function_name}")
        target = function_name
        open("debug.txt", "w").write(function_name)
    else:
        open("debug.txt", "w").write(completion['choices'][0]['message']['content'])

    history_file = load_conversation_history(conversation_id)

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["related_documents"] = query_db(user_message)
    context["chat_history"] = get_chat_history(conversation_id)

    # chain 별로 db 분리하면 좋을듯.
    target_chain = default_chain
    if target == "kakao_sink":
        target_chain = default_chain
    elif target == "kakao_social":
        target_chain = default_chain

    answer = target_chain.run(context)

    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)
    return {"answer": answer}

class Message(Base):
    original_text: str
    text: str
    created_at: str


# class Training(pc.State):




class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []

    # @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Translations will appear here."
        translated = generate_answer(self.text)["answer"]
        return translated

    def post(self):
        self.messages = [
            Message(
                original_text=self.text,
                text=self.output(),
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
            )
        ] + self.messages


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Kakao Helper bot 🗺", font_size="2rem"),
        pc.text(
            "Hello! I am kakao helper bot!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.original_text),
            down_arrow(),
            text_box(message.text),
            pc.box(
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )

def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="Text to input",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Kakao Helper Bot")
app.compile()
