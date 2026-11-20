"""LangChain adapter: use any lab model (base or LoRA version) as a LangChain chat model.

    from lab.lc import LabChatModel
    llm = LabChatModel(model_id="sft-001")
    (ChatPromptTemplate.from_messages([...]) | llm | StrOutputParser()).invoke({...})
"""
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from . import engines

_ROLE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def to_dicts(messages: List[BaseMessage]):
    return [{"role": _ROLE.get(m.type, "user"), "content": m.content} for m in messages]


class LabChatModel(BaseChatModel):
    model_id: str
    max_new_tokens: int = 256
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "ai-training-lab"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        text = engines.chat(self.model_id, to_dicts(messages), max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
