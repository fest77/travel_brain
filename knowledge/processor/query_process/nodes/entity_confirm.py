"""实体确认节点：识别用户问题中的旅游目的地与内容意图

流程（分步注释）:
    Step1 取历史对话 → 拼上下文
    Step2 LLM 提取 region/content_type + 改写独立问题(含多轮指代消解)
    Step3 解析/清洗 JSON，失败则降级为原文
    Step4 回填 state：entity_names(目的地)、rewritten_query、history
"""
import json
import re
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.prompt.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT, ENTITY_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import get_recent_messages


class EntityConfirmNode(BaseNode):
    """从问题(+历史)中提取旅游目的地/意图，改写为可检索的独立问题"""

    name = "entity_confirm"

    def process(self, state):
        # Step1 获取历史对话（最近若干条，供多轮指代消解）
        session_id = state.get("session_id", "")
        original_query = state.get("original_query", "")
        history_messages = get_recent_messages(session_id, limit=10)
        history_messages.reverse()  # 时间升序便于 LLM 阅读

        history_text = "暂无历史对话信息"
        if history_messages:
            history_text = "\n".join(
                f"{m.get('role')}: {m.get('text')}" for m in history_messages if m.get("text")
            )

        # Step2 LLM 提取 目的地/内容类型 + 改写
        region, content_type, rewritten = self._extract(original_query, history_text)

        # Step3 决策回填
        entity_names: List[str] = [region] if region else []
        self.logger.info(f"实体确认: region={region}, content_type={content_type}, rewritten={rewritten[:50]}...")

        state["entity_names"] = entity_names
        state["rewritten_query"] = rewritten or original_query
        state["history"] = history_messages
        return state

    # ------------------------------------------------------------ #
    def _extract(self, query: str, history_text: str):
        """调用 LLM 提取；任何失败都降级：region/content_type 空，rewritten=原文"""
        default = ("", "", query)
        try:
            llm_client = AIClients.get_llm_openai(response_format=True)
            user_prompt = ENTITY_EXTRACT_TEMPLATE.format(query=query, history_text=history_text)
            resp = llm_client.invoke([
                SystemMessage(content=ENTITY_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            text = (resp.content or "").strip()
            if not text:
                return default
            data = self._parse_json(text)
            region = str(data.get("region") or "").strip()
            content_type = str(data.get("content_type") or "").strip()
            rewritten = str(data.get("rewritten_query") or "").strip()
            return region, content_type, rewritten or query
        except Exception as e:
            self.logger.warning(f"实体提取失败，降级为原文: {e}")
            return default

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """去掉 ```json 围栏后解析 JSON，容错"""
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        # 找到第一个 { 到最后一个 }（防御模型输出多余文字）
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)
