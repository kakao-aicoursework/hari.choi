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


openai.api_key = open("../appkey.txt", "r").read()
os.environ["OPENAI_API_KEY"] = open("../appkey.txt", "r").read()
llm = ChatOpenAI(temperature=0.8)
info = open("project_data_카카오싱크.txt", "r").read()

def call_gpt(text) -> str:
    global info

    system_instruction = (f"assistant는 정보를 제공하는 앱으로서 동작한다."
                          f"너의 이름은 카카오 헬퍼봇이야. 너가 알고 있는 정보는 다음과 같아."
                          f"--------------------"
                          f"{info}")

    chat_prompt = ChatPromptTemplate.from_messages([SystemMessage(content=system_instruction),
                                                    HumanMessagePromptTemplate.from_template("{text}\n---\n위의 내용에 응답해줘")])

    return LLMChain(llm=llm,prompt=chat_prompt).run(text=text)


class Message(Base):
    original_text: str
    text: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Translations will appear here."
        translated = call_gpt(self.text)
        return translated

    def post(self):
        self.messages = [
            Message(
                original_text=self.text,
                text=self.output,
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
