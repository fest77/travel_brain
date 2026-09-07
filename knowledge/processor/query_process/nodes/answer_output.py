"""答案生成节点：基于精排片段组装 prompt → LLM 生成 → 写历史 → 保存结果

支持流式(is_stream=True，SSE 推 DELTA)与非流式两种。
"""
from typing import Dict, List, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.prompt.query_prompt import ANSWER_PROMPT, ONLINE_ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    """生成并输出最终答案（含来源引用 + 历史落库）"""

    name = "answer_output"

    def process(self, state):
        # Step1 判断本地检索结果是否"足够相关"
        #      · 无参考片段 或 最高相关分低于阈值(如问北京等库外目的地)
        #        → 判定本地无法回答，降级为「实时联网」兜底
        docs = state.get("reranked_docs") or []
        top_score = docs[0].get("score") if docs and docs[0].get("score") is not None else 0.0
        use_local = len(docs) > 0 and float(top_score) >= self.config.min_local_score
        self.logger.info(f"本地参考判断: 片段数={len(docs)}, 最高分={top_score:.4f}, "
                         f"使用本地={use_local}(阈值={self.config.min_local_score})")

        if not use_local:
            # Step2a 本地无资料 → 用原生 OpenAI + enable_search(实时联网) 回答
            self._generate_answer_online(state)
        else:
            # Step2b 本地有资料 → 组装「基于参考内容」prompt 后常规生成
            prompt = self._build_prompt(state)
            self._generate_answer(state, prompt)

        # Step3 写入历史对话
        self._write_history(state)
        return state

    # ------------------------------------------------------------ #
    def _build_prompt(self, state) -> str:
        question = state.get("rewritten_query") or state.get("original_query", "")
        docs = state.get("reranked_docs") or []
        char_budget = self.config.max_context_chars

        context, budget = self._format_docs(docs, char_budget)
        history, _ = self._format_history(state.get("history") or [], budget)

        entity = ", ".join(state.get("entity_names") or []) or "未指定"
        content_type = state.get("content_type") if state.get("content_type") else ""
        if content_type:
            entity = f"{entity}({content_type})" if entity != "未指定" else content_type

        return ANSWER_PROMPT.format(
            context=context or "无参考内容",
            history=history or "暂无历史对话",
            entity=entity,
            question=question,
        )

    def _format_docs(self, docs: List[Dict], budget: int) -> Tuple[str, int]:
        """把精排片段格式化为带引用标签的参考文本 [来源] [出处]"""
        lines, used = [], 0
        for i, doc in enumerate(docs, 1):
            content = (doc.get("content") or "").strip()
            if not content:
                continue
            tags = [f"[{i}]", f"[来源:{doc.get('source', '')}]"]
            if doc.get("url"):
                tags.append(f"[url:{doc['url']}]")
            if doc.get("chunk_id") is not None:
                tags.append(f"[chunk_id:{doc.get('chunk_id')}]")
            if doc.get("region"):
                tags.append(f"[目的地:{doc.get('region')}]")
            if doc.get("content_type"):
                tags.append(f"[类型:{doc.get('content_type')}]")
            title = doc.get("title") or doc.get("file_title") or ""
            entry = " ".join(tags) + "\n" + (f"标题:{title}\n" if title else "") + content
            if used + len(entry) > budget:
                break
            lines.append(entry)
            used += len(entry) + 2
        return "\n\n".join(lines), budget - used

    def _format_history(self, history: List[Dict], budget: int) -> Tuple[str, int]:
        lines, used = [], 0
        labels = {"user": "用户", "assistant": "助手"}
        for m in history:
            role = labels.get(m.get("role", ""))
            text = m.get("text", "")
            if not role or not text:
                continue
            line = f"{role}: {text}"
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines), budget - used

    def _generate_answer_online(self, state):
        """本地无资料：原生 OpenAI 客户端 + extra_body['enable_search'] 触发 DashScope 实时联网。

        说明：ChatOpenAI 传 model_kwargs 不一定触发 enable_search，这里用原生 SDK 方式(已实测有效)。
        """
        question = state.get("rewritten_query") or state.get("original_query", "")
        history, _ = self._format_history(state.get("history") or [], 3000)
        user_text = ONLINE_ANSWER_PROMPT.format(
            history=history or "暂无历史对话",
            question=question,
        )
        is_stream = bool(state.get("is_stream"))
        task_id = state.get("task_id")
        try:
            client = AIClients.get_openai()
            params = dict(
                model=self.config.default_model,
                messages=[{"role": "user", "content": user_text}],
                extra_body={"enable_search": True},   # 关键：DashScope 实时联网开关
            )
            if is_stream:
                parts = []
                stream = client.chat.completions.create(stream=True, **params)
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content or ""
                    parts.append(delta)
                    push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": delta})
                state["answer"] = "".join(parts)
            else:
                resp = client.chat.completions.create(**params)
                state["answer"] = (resp.choices[0].message.content or "").strip()
            set_task_result(task_id, "answer", state["answer"])
        except Exception as e:
            self.logger.error(f"实时联网回答失败: {e}")
            state["answer"] = f"抱歉，实时联网回答失败：{e}"
            set_task_result(task_id, "answer", state["answer"])

    # ------------------------------------------------------------ #
    def _generate_answer(self, state, prompt):
        """本地有资料时的常规生成（流式走 SSE，非流式 set_task_result）"""
        try:
            llm_client = AIClients.get_llm_openai(response_format=False)
        except Exception as e:
            state["answer"] = f"抱歉，LLM 客户端不可用，无法生成答案：{e}"
            set_task_result(state.get("task_id"), "answer", state["answer"])
            return

        if state.get("is_stream"):
            answer_parts = []
            try:
                for chunk in llm_client.stream(prompt):
                    delta = chunk.content or ""
                    answer_parts.append(delta)
                    push_sse_event(task_id=state.get("task_id"), event=SSEEvent.DELTA,
                                   data={"delta": delta})
            except Exception as e:
                self.logger.warning(f"流式生成失败: {e}")
            state["answer"] = "".join(answer_parts)
        else:
            try:
                resp = llm_client.invoke(prompt)
                answer = (resp.content or "").strip()
            except Exception as e:
                answer = f"抱歉，生成答案失败：{e}"
            state["answer"] = answer
            set_task_result(state.get("task_id"), "answer", answer)

    def _write_history(self, state):
        """把这一轮问答写入 MongoDB 历史（user + assistant 两条）"""
        try:
            sid = state.get("session_id")
            original = state.get("original_query", "")
            rewritten = state.get("rewritten_query") or original
            entity_names = state.get("entity_names") or []
            # 用户问题
            save_chat_message(session_id=sid, role="user", text=original,
                              rewritten_query=rewritten, entity_names=entity_names)
            # 助手答案
            save_chat_message(session_id=sid, role="assistant", text=state.get("answer", ""),
                              rewritten_query=rewritten, entity_names=entity_names)
        except Exception as e:
            self.logger.error(f"写入历史失败: {e}")
