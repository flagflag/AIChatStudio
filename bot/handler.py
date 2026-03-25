import base64
import json
import logging
import re
import threading
import time
import uuid

import httpx
import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetMessageResourceRequest,
    ListMessageRequest,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from providers.base import AIProvider
from bot.session import SessionManager
from bot.agent_manager import AgentManager

logger = logging.getLogger(__name__)


def _build_approval_card(request_id: str, user_id: str, text: str,
                         chat_name: str = "", chat_id: str = "") -> str:
    """构建审批卡片 JSON。"""
    # 截断过长的消息预览
    preview = text if len(text) <= 200 else text[:200] + "..."
    elements = []
    if chat_name or chat_id:
        link = f"https://applink.feishu.cn/client/chat/open?openChatId={chat_id}" if chat_id else ""
        source_text = f"**来源群聊：** {chat_name or chat_id}"
        if link:
            source_text += f"  [打开群聊]({link})"
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": source_text},
        })
    elements.append({
        "tag": "div",
        "text": {"tag": "lark_md", "content": f"**发送者：** <at id={user_id}></at>"},
    })
    elements.append({
        "tag": "div",
        "text": {"tag": "lark_md", "content": f"**消息内容：**\n{preview}"},
    })
    elements.append({"tag": "hr"})
    elements.append({
        "tag": "action",
        "actions": [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "✅ 批准执行"},
                "type": "primary",
                "value": {"action": "approve", "request_id": request_id},
            },
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "❌ 拒绝"},
                "type": "danger",
                "value": {"action": "reject", "request_id": request_id},
            },
        ],
    })
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "🔒 Agent 执行审批"},
            "template": "orange",
        },
        "elements": elements,
    }
    return json.dumps(card)

# 命令 -> provider key 的映射
COMMAND_MAP = {
    "/claude": "claude",
    "/gemini": "gemini",
    "/gpt": "openai",
}

HELP_TEXT = """🤖 AI 多模型机器人

**切换模型命令：**
• /claude <消息> — 使用 Claude
• /gemini <消息> — 使用 Gemini
• /gpt <消息> — 使用 OpenAI GPT

**其他命令：**
• /model — 查看当前默认模型
• /clear — 清除对话上下文
• /help — 显示此帮助

**直接发消息（无命令前缀）将使用默认模型。**
**每条新消息会自动创建话题，话题内对话由独立 Agent 处理。**"""


class MessageHandler:

    def __init__(
        self,
        providers: dict[str, AIProvider],
        default_provider: str,
        session_manager: SessionManager,
        lark_client: lark.Client,
        agent_manager: AgentManager,
        bot_open_id: str = "",
        feishu_cfg: dict | None = None,
        admin_ids: list[str] | None = None,
        approval_expire_minutes: int = 0,
    ):
        self.providers = providers
        self.default_provider = default_provider
        self.session_manager = session_manager
        self.lark_client = lark_client
        self.agent_manager = agent_manager
        self.bot_open_id = bot_open_id
        self._feishu_cfg = feishu_cfg or {}
        self._admin_ids = admin_ids or []
        self._approval_expire = approval_expire_minutes * 60
        self._processed_msgs: dict[str, float] = {}
        # 待审批请求：request_id -> {message_id, text, images, user_id, timestamp}
        self._pending_requests: dict[str, dict] = {}
        # 已批准用户：user_open_id -> 批准时间戳
        self._approved_users: dict[str, float] = {}
        # 话题消息历史：thread_id -> [{user_id, text, is_bot_mention, timestamp}]
        self._thread_messages: dict[str, list[dict]] = {}
        # 用户名缓存：open_id -> 名字
        self._user_name_cache: dict[str, str] = {}

    def handle(self, event_data: dict):
        """处理飞书消息事件"""
        message = event_data.get("message", {})
        message_id = message.get("message_id", "")
        chat_id = message.get("chat_id", "")
        chat_type = message.get("chat_type", "")
        thread_id = message.get("thread_id", "")
        mentions = message.get("mentions", [])
        user_id = event_data.get("sender", {}).get("sender_id", {}).get("open_id", "")

        logger.debug("收到消息: chat_type=%s, thread_id=%s, mentions=%s, user_id=%s, bot_open_id=%s",
                     chat_type, thread_id, mentions, user_id, self.bot_open_id)

        # 机器人自己的消息：仅存储话题历史，不处理
        if self.bot_open_id and user_id == self.bot_open_id:
            if thread_id:
                bot_text = self._extract_text_quick(message)
                if bot_text:
                    self._store_thread_message(thread_id, user_id, bot_text, is_bot_mention=False)
            return

        # 消息去重
        if message_id in self._processed_msgs:
            return
        self._processed_msgs[message_id] = time.time()
        cutoff = time.time() - 300
        self._processed_msgs = {k: v for k, v in self._processed_msgs.items() if v > cutoff}

        msg_type = message.get("message_type", "")
        if msg_type not in ("text", "post", "image"):
            return

        try:
            content = json.loads(message.get("content", "{}"))
        except json.JSONDecodeError:
            return

        # 解析消息内容
        text = ""
        images: list[dict] = []  # [{"media_type": "image/png", "data": "base64..."}]

        if msg_type == "text":
            text = content.get("text", "").strip()
        elif msg_type == "post":
            text, image_keys = self._extract_post_content(content)
            for ik in image_keys:
                img = self._download_image(message_id, ik)
                if img:
                    images.append(img)
        elif msg_type == "image":
            image_key = content.get("image_key", "")
            if image_key:
                img = self._download_image(message_id, image_key)
                if img:
                    images.append(img)

        # 判断是否 @了机器人
        is_bot_mentioned = True  # 私聊默认触发
        if chat_type == "group":
            is_bot_mentioned = any(
                getattr(getattr(m, "id", None), "open_id", None) == self.bot_open_id
                for m in mentions
            ) if self.bot_open_id else bool(mentions)

        # 清理 @提及标记
        clean_text = re.sub(r"@_user_\d+\s*", "", text).strip()

        # 话题内消息：无论是否 @机器人 都存储（用于上下文提取）
        if thread_id:
            self._store_thread_message(thread_id, user_id, clean_text, is_bot_mentioned)
            if not is_bot_mentioned:
                return  # 存储后直接返回，不处理

        # 群聊非话题消息需要 @机器人 才触发
        if chat_type == "group" and not is_bot_mentioned:
            return

        text = clean_text
        if not text and not images:
            return

        # 只有图片没有文字也允许继续（agent_manager 会处理）
        if not text and images:
            text = ""

        # 话题内消息 → 需要审批 or 路由到 Agent
        if thread_id:
            if self._admin_ids and user_id not in self._admin_ids and not self._is_user_approved(user_id):
                self._send_thread_approval_request(thread_id, message_id, chat_id, text, images, user_id)
            else:
                self._handle_agent_message(thread_id, text, message_id, images)
            return

        # 话题外的命令处理
        if text == "/help":
            self._reply(chat_id, HELP_TEXT)
            return

        if text == "/myid":
            self._reply(chat_id, f"你的 open_id: `{user_id}`")
            return

        if text == "/model":
            provider = self.providers.get(self.default_provider)
            name = provider.name if provider else self.default_provider
            self._reply(chat_id, f"当前默认模型: **{name}**")
            return

        if text == "/clear":
            self.session_manager.clear(chat_id, user_id)
            self._reply(chat_id, "对话上下文已清除 ✅")
            return

        for cmd, key in COMMAND_MAP.items():
            if text.startswith(cmd):
                if key not in self.providers:
                    self._reply(chat_id, f"模型 **{key}** 未配置，请检查 config.yaml")
                    return
                user_text = text[len(cmd):].strip()
                if not user_text:
                    self._reply(chat_id, "请输入消息内容")
                    return
                # 模型切换命令走原有 session 逻辑
                self._handle_provider_chat(key, chat_id, user_id, user_text)
                return

        # 普通消息 → 管理员/已批准用户直接执行，其他人需要审批
        if self._admin_ids and user_id not in self._admin_ids and not self._is_user_approved(user_id):
            self._send_approval_request(message_id, chat_id, text, images, user_id)
        else:
            threading.Thread(
                target=self._handle_new_agent_thread,
                args=(message_id, text, images),
                daemon=True,
            ).start()

    def _extract_text_quick(self, message: dict) -> str:
        """轻量提取消息文本（不下载图片），用于存储机器人自己的消息。"""
        msg_type = message.get("message_type", "")
        try:
            content = json.loads(message.get("content", "{}"))
        except json.JSONDecodeError:
            return ""
        if msg_type == "text":
            return content.get("text", "").strip()
        elif msg_type == "post":
            text, _ = self._extract_post_content(content)
            return text
        return ""

    def _store_thread_message(self, thread_id: str, user_id: str, text: str, is_bot_mention: bool):
        """存储话题内的消息，用于上下文提取。"""
        if thread_id not in self._thread_messages:
            self._thread_messages[thread_id] = []
        self._thread_messages[thread_id].append({
            "user_id": user_id,
            "text": text,
            "is_bot_mention": is_bot_mention,
            "timestamp": time.time(),
        })
        # 每个话题最多保留 200 条
        if len(self._thread_messages[thread_id]) > 200:
            self._thread_messages[thread_id] = self._thread_messages[thread_id][-200:]

    def _get_thread_context(self, thread_id: str) -> str:
        """提取话题上下文。

        - Agent 不存在（新启动）：通过飞书 API 拉取完整话题历史（含 AI 回复）。
        - Agent 已存在：返回上一次 @机器人 到当前之间的其他用户对话。
        """
        if not self.agent_manager.has_session(thread_id):
            # 新 Agent → 通过飞书 API 拉取话题历史
            return self._fetch_thread_history(thread_id)

        # 已有 Agent → 取两次 @机器人 之间的对话（本地存储）
        messages = self._thread_messages.get(thread_id, [])
        if len(messages) < 2:
            return ""

        context_msgs = []
        for i in range(len(messages) - 2, -1, -1):
            msg = messages[i]
            if msg["is_bot_mention"]:
                break
            # 跳过机器人自己的消息（Agent 已有记忆）
            if self.bot_open_id and msg["user_id"] == self.bot_open_id:
                continue
            context_msgs.insert(0, msg)

        if not context_msgs:
            return ""

        lines = []
        for msg in context_msgs:
            name = self._get_user_name(msg["user_id"])
            lines.append(f"{name}: {msg['text']}")

        return "[话题上下文 - 以下是上次 @机器人 之后其他用户的对话]\n" + "\n".join(lines)

    def _fetch_thread_history(self, thread_id: str) -> str:
        """通过飞书 API 拉取话题历史消息（用于新 Agent 启动时获取完整上下文）。"""
        try:
            all_lines = []
            page_token = None

            while True:
                builder = ListMessageRequest.builder() \
                    .container_id_type("thread") \
                    .container_id(thread_id) \
                    .sort_type("ByCreateTimeAsc") \
                    .page_size(50)
                if page_token:
                    builder = builder.page_token(page_token)

                request = builder.build()
                resp = self.lark_client.im.v1.message.list(request)

                if not resp.success():
                    logger.error("拉取话题历史失败: %s - %s", resp.code, resp.msg)
                    return ""

                if resp.data and resp.data.items:
                    for msg in resp.data.items:
                        # 跳过已删除的消息
                        if msg.deleted:
                            continue

                        msg_type = msg.msg_type or ""
                        # 跳过不支持的消息类型
                        if msg_type not in ("text", "post"):
                            continue

                        # 提取发送者
                        sender_id = ""
                        if msg.sender:
                            sender_id = msg.sender.id or ""

                        # 提取文本
                        text = ""
                        try:
                            content = json.loads(msg.body.content or "{}")
                            if msg_type == "text":
                                text = content.get("text", "")
                            elif msg_type == "post":
                                text, _ = self._extract_post_content(content)
                        except (json.JSONDecodeError, AttributeError):
                            continue

                        if not text:
                            continue

                        # 清理 @提及标记
                        text = re.sub(r"@_user_\d+\s*", "", text).strip()
                        if not text:
                            continue

                        # 确定显示名称
                        if self.bot_open_id and sender_id == self.bot_open_id:
                            name = "AI助手"
                        else:
                            name = self._get_user_name(sender_id)

                        # 截断过长的单条消息
                        if len(text) > 500:
                            text = text[:500] + "..."

                        all_lines.append(f"{name}: {text}")

                if not resp.data or not resp.data.has_more:
                    break
                page_token = resp.data.page_token

            if not all_lines:
                return ""

            # 排除最后一条（当前用户的查询消息）
            if len(all_lines) > 1:
                all_lines = all_lines[:-1]
            else:
                return ""

            return "[话题历史 - Agent 重新启动，以下是之前的完整对话]\n" + "\n".join(all_lines)

        except Exception:
            logger.exception("拉取话题历史异常")
            return ""

    def _is_user_approved(self, user_id: str) -> bool:
        """检查用户是否在批准有效期内。"""
        if not self._approval_expire or user_id not in self._approved_users:
            return False
        elapsed = time.time() - self._approved_users[user_id]
        if elapsed < self._approval_expire:
            return True
        del self._approved_users[user_id]
        return False

    def _create_pending_request(self, message_id: str, chat_id: str, text: str,
                                images: list[dict], user_id: str, **extra) -> str:
        """创建待审批请求，返回 request_id。"""
        request_id = uuid.uuid4().hex[:12]
        self._pending_requests[request_id] = {
            "message_id": message_id,
            "chat_id": chat_id,
            "text": text,
            "images": images,
            "user_id": user_id,
            "timestamp": time.time(),
            **extra,
        }
        # 清理超过 30 分钟的过期请求
        cutoff = time.time() - 1800
        self._pending_requests = {
            k: v for k, v in self._pending_requests.items()
            if v["timestamp"] > cutoff
        }
        return request_id

    def _send_approval_card(self, request_id: str, user_id: str, text: str, chat_id: str):
        """私信发送审批卡片给所有管理员。"""
        chat_name = self._get_chat_name(chat_id)
        card_json = _build_approval_card(request_id, user_id, text, chat_name, chat_id)
        for admin_id in self._admin_ids:
            body = CreateMessageRequestBody.builder() \
                .receive_id(admin_id) \
                .msg_type("interactive") \
                .content(card_json) \
                .build()
            request = CreateMessageRequest.builder() \
                .receive_id_type("open_id") \
                .request_body(body) \
                .build()
            resp = self.lark_client.im.v1.message.create(request)
            if not resp.success():
                logger.error("发送审批卡片给 %s 失败: %s - %s", admin_id, resp.code, resp.msg)

    def _send_approval_request(self, message_id: str, chat_id: str, text: str,
                                images: list[dict], user_id: str):
        """话题外新消息的审批流程。"""
        request_id = self._create_pending_request(
            message_id, chat_id, text, images, user_id, source="new")

        # 在话题里回复提示，记录话题信息供审批通过后使用
        resp = self._reply_in_thread(message_id, "🔒 该请求需要管理员审批，请稍候...")
        hint_msg_id = None
        thread_id = None
        if resp and resp.success() and resp.data:
            hint_msg_id = getattr(resp.data, "message_id", None)
            thread_id = getattr(resp.data, "thread_id", None)
        self._pending_requests[request_id]["hint_msg_id"] = hint_msg_id
        self._pending_requests[request_id]["thread_id"] = thread_id

        self._send_approval_card(request_id, user_id, text, chat_id)

    def _send_thread_approval_request(self, thread_id: str, message_id: str,
                                       chat_id: str, text: str,
                                       images: list[dict], user_id: str):
        """话题内消息的审批流程。"""
        request_id = self._create_pending_request(
            message_id, chat_id, text, images, user_id,
            source="thread", thread_id=thread_id)

        # 在话题内回复提示
        resp = self._reply_to_message(message_id, "🔒 该请求需要管理员审批，请稍候...")
        hint_msg_id = None
        if resp and resp.success() and resp.data:
            hint_msg_id = getattr(resp.data, "message_id", None)
        self._pending_requests[request_id]["hint_msg_id"] = hint_msg_id

        self._send_approval_card(request_id, user_id, text, chat_id)

    def handle_card_action(self, action_value: dict, operator_id: str) -> dict:
        """处理卡片按钮点击事件。返回 {toast, card?}。"""
        request_id = action_value.get("request_id", "")
        action = action_value.get("action", "")

        # 权限检查
        if operator_id not in self._admin_ids:
            logger.warning("非管理员尝试审批: %s", operator_id)
            return {"toast": {"type": "info", "content": "⚠️ 你没有审批权限"}}

        pending = self._pending_requests.pop(request_id, None)
        if not pending:
            return {"toast": {"type": "info", "content": "⚠️ 该请求已过期或已处理"}}

        if action == "approve":
            logger.info("管理员 %s 批准了请求 %s", operator_id, request_id)
            # 记录该用户已被批准
            self._approved_users[pending["user_id"]] = time.time()
            # 根据来源选择处理方式
            if pending.get("source") == "thread":
                threading.Thread(
                    target=self._handle_approved_thread_message,
                    args=(pending,),
                    daemon=True,
                ).start()
            else:
                threading.Thread(
                    target=self._handle_approved_agent,
                    args=(pending,),
                    daemon=True,
                ).start()
            return {
                "toast": {"type": "success", "content": "已批准，Agent 正在启动"},
                "card": self._build_result_card("✅ Agent 执行已批准", "green",
                    f"**审批人：** <at id={operator_id}></at>\n**状态：** 已批准，Agent 正在执行"),
            }
        else:
            logger.info("管理员 %s 拒绝了请求 %s", operator_id, request_id)
            # 在话题内通知拒绝
            hint_msg_id = pending.get("hint_msg_id")
            if hint_msg_id:
                self._reply_to_message(hint_msg_id, "❌ 管理员已拒绝该请求")
            return {
                "toast": {"type": "info", "content": "已拒绝"},
                "card": self._build_result_card("❌ Agent 执行已拒绝", "red",
                    f"**审批人：** <at id={operator_id}></at>\n**状态：** 已拒绝"),
            }

    @staticmethod
    def _build_result_card(title: str, template: str, content: str) -> dict:
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": template,
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": content}},
            ],
        }

    def _handle_provider_chat(self, provider_key: str, chat_id: str, user_id: str, text: str):
        """使用指定 provider 进行对话（原有逻辑）。"""
        session = self.session_manager.get(chat_id, user_id)
        session.add_user_message(text)

        provider = self.providers[provider_key]
        try:
            reply = provider.chat(session.get_messages())
            session.add_assistant_message(reply)
        except Exception as e:
            logger.exception("AI 调用失败")
            reply = f"调用 {provider.name} 失败: {e}"
            if session.messages and session.messages[-1]["role"] == "user":
                session.messages.pop()

        self._reply(chat_id, reply)

    def _handle_approved_thread_message(self, pending: dict):
        """审批通过后，在已有话题中继续 Agent 对话。"""
        thread_id = pending["thread_id"]
        message_id = pending["message_id"]
        text = pending["text"]
        images = pending.get("images")
        hint_msg_id = pending.get("hint_msg_id")

        if hint_msg_id:
            self._reply_to_message(hint_msg_id, "🤖 审批已通过，正在处理中...")

        # 拼接上下文
        context = self._get_thread_context(thread_id)
        prompt = text
        if context:
            prompt = f"{context}\n\n[用户消息]\n{text}"

        try:
            reply = self.agent_manager.chat(thread_id, prompt, images=images)
        except Exception as e:
            logger.exception("Agent 调用失败")
            reply = f"Agent 调用失败: {e}"

        target_id = hint_msg_id or message_id
        resp = self._reply_to_message(target_id, reply)

        reply_msg_id = None
        if resp and resp.success() and resp.data:
            reply_msg_id = getattr(resp.data, "message_id", None)
        self.agent_manager.set_reply_message_id(thread_id, reply_msg_id or target_id)

    def _handle_approved_agent(self, pending: dict):
        """审批通过后，在已有话题中启动 Agent。"""
        hint_msg_id = pending.get("hint_msg_id")
        thread_id = pending.get("thread_id")
        message_id = pending["message_id"]
        text = pending["text"]
        images = pending.get("images")

        # 在话题内发送启动提示
        if hint_msg_id:
            self._reply_to_message(hint_msg_id, "🤖 审批已通过，Agent 正在处理中...")

        thread_key = thread_id or f"pending:{message_id}"
        try:
            reply = self.agent_manager.chat(thread_key, text, images=images)
        except Exception as e:
            logger.exception("Agent 调用失败")
            reply = f"Agent 调用失败: {e}"

        # 回复到话题内
        target_id = hint_msg_id or message_id
        resp = self._reply_to_message(target_id, reply)

        reply_msg_id = None
        if resp and resp.success() and resp.data:
            reply_msg_id = getattr(resp.data, "message_id", None)
        self.agent_manager.set_reply_message_id(thread_key, reply_msg_id or target_id)

    def _handle_new_agent_thread(self, message_id: str, text: str, images: list[dict] | None = None):
        """创建新的飞书话题并启动 Agent 会话。"""
        # 先创建话题，发送"处理中"提示
        resp = self._reply_in_thread(message_id, "🤖 Agent 已启动，正在处理中...")
        # 获取"处理中"消息的 message_id，后续回复都 reply 到这条消息（保持在话题内）
        hint_msg_id = None
        thread_id = None
        if resp and resp.success() and resp.data:
            hint_msg_id = getattr(resp.data, "message_id", None)
            thread_id = getattr(resp.data, "thread_id", None)

        thread_key = thread_id or f"pending:{message_id}"
        try:
            reply = self.agent_manager.chat(thread_key, text, images=images)
        except Exception as e:
            logger.exception("Agent 调用失败")
            reply = f"Agent 调用失败: {e}"

        # 回复到话题内（reply 到"处理中"那条消息）
        if hint_msg_id:
            resp = self._reply_to_message(hint_msg_id, reply)
        else:
            resp = self._reply_in_thread(message_id, reply)

        # 记录最新的 reply_message_id，供清理通知使用
        reply_msg_id = None
        if resp and resp.success() and resp.data:
            reply_msg_id = getattr(resp.data, "message_id", None)
        self.agent_manager.set_reply_message_id(thread_key, reply_msg_id or hint_msg_id or message_id)

    def _handle_agent_message(self, thread_id: str, text: str, message_id: str, images: list[dict] | None = None):
        """在已有话题内继续 Agent 对话（异步执行）。"""
        # 提取上下文（在启动线程前，避免竞态）
        context = self._get_thread_context(thread_id)

        def _run():
            # 先发提示
            resp = self._reply_to_message(message_id, "🤖 正在思考中...")
            hint_msg_id = None
            if resp and resp.success() and resp.data:
                hint_msg_id = getattr(resp.data, "message_id", None)

            # 拼接上下文
            prompt = text
            if context:
                prompt = f"{context}\n\n[用户消息]\n{text}"

            try:
                reply = self.agent_manager.chat(thread_id, prompt, images=images)
            except Exception as e:
                logger.exception("Agent 调用失败")
                reply = f"Agent 调用失败: {e}"

            # 回复到话题内
            target_id = hint_msg_id or message_id
            resp = self._reply_to_message(target_id, reply)

            # 更新 reply_message_id
            reply_msg_id = None
            if resp and resp.success() and resp.data:
                reply_msg_id = getattr(resp.data, "message_id", None)
            self.agent_manager.set_reply_message_id(thread_id, reply_msg_id or target_id)

        threading.Thread(target=_run, daemon=True).start()

    def _download_image(self, message_id: str, image_key: str) -> dict | None:
        """从飞书下载图片，返回 {"media_type": ..., "data": base64} 或 None。"""
        try:
            request = GetMessageResourceRequest.builder() \
                .message_id(message_id) \
                .file_key(image_key) \
                .type("image") \
                .build()
            resp = self.lark_client.im.v1.message_resource.get(request)
            if resp.success():
                img_bytes = resp.file.read()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                # 简单判断图片格式
                media_type = "image/png"
                if img_bytes[:3] == b'\xff\xd8\xff':
                    media_type = "image/jpeg"
                elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
                    media_type = "image/webp"
                elif img_bytes[:3] == b'GIF':
                    media_type = "image/gif"
                logger.info("下载图片成功: %s (%d bytes)", image_key, len(img_bytes))
                return {"media_type": media_type, "data": b64}
            else:
                logger.error("下载图片失败: %s - %s", resp.code, resp.msg)
        except Exception:
            logger.exception("下载图片异常: %s", image_key)
        return None

    def _get_user_name(self, open_id: str) -> str:
        """获取用户名（带缓存）。"""
        if open_id in self._user_name_cache:
            return self._user_name_cache[open_id]
        try:
            from lark_oapi.api.contact.v3 import GetUserRequest
            request = GetUserRequest.builder().user_id(open_id).user_id_type("open_id").build()
            resp = self.lark_client.contact.v3.user.get(request)
            if resp.success() and resp.data and resp.data.user:
                name = resp.data.user.name or open_id
                self._user_name_cache[open_id] = name
                return name
        except Exception:
            logger.debug("获取用户名失败: %s", open_id)
        self._user_name_cache[open_id] = open_id
        return open_id

    def _get_chat_name(self, chat_id: str) -> str:
        """获取群聊名称。"""
        try:
            from lark_oapi.api.im.v1 import GetChatRequest
            request = GetChatRequest.builder().chat_id(chat_id).build()
            resp = self.lark_client.im.v1.chat.get(request)
            if resp.success() and resp.data:
                return resp.data.name or ""
            else:
                logger.warning("获取群聊名称失败: %s - %s", resp.code, resp.msg)
        except Exception:
            logger.exception("获取群聊名称异常: %s", chat_id)
        return ""

    @staticmethod
    def _extract_post_content(content: dict) -> tuple[str, list[str]]:
        """从飞书 post（富文本）消息中提取纯文本和图片 key。"""
        parts = []
        image_keys = []
        # post 结构: {"title": "...", "content": [[{tag, text}, ...], ...]}
        # 多语言时外层有 zh_cn/en_us 等 key
        post_body = content
        if "content" not in post_body:
            # 多语言格式：取第一个语言
            for lang_key in post_body:
                if isinstance(post_body[lang_key], dict):
                    post_body = post_body[lang_key]
                    break

        title = post_body.get("title", "")
        if title:
            parts.append(title)

        for line in post_body.get("content", []):
            line_texts = []
            for element in line:
                tag = element.get("tag", "")
                if tag == "text":
                    line_texts.append(element.get("text", ""))
                elif tag == "a":
                    line_texts.append(element.get("text", ""))
                elif tag == "at":
                    pass  # @提及，跳过
                elif tag == "md":
                    line_texts.append(element.get("text", ""))
                elif tag == "img":
                    ik = element.get("image_key", "")
                    if ik:
                        image_keys.append(ik)
            if line_texts:
                parts.append("".join(line_texts))

        return "\n".join(parts).strip(), image_keys

    def _reply(self, chat_id: str, text: str):
        """向飞书群聊发送消息"""
        body = CreateMessageRequestBody.builder() \
            .receive_id(chat_id) \
            .msg_type("text") \
            .content(json.dumps({"text": text})) \
            .build()

        request = CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(body) \
            .build()

        response = self.lark_client.im.v1.message.create(request)
        if not response.success():
            logger.error(f"发送消息失败: {response.code} - {response.msg}")
        return response

    def _reply_in_thread(self, message_id: str, text: str):
        """以话题形式回复消息（创建新话题）"""
        body = ReplyMessageRequestBody.builder() \
            .msg_type("text") \
            .content(json.dumps({"text": text})) \
            .reply_in_thread(True) \
            .build()

        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(body) \
            .build()

        response = self.lark_client.im.v1.message.reply(request)
        if not response.success():
            logger.error(f"话题回复失败: {response.code} - {response.msg}")
        return response

    def _reply_to_message(self, message_id: str, text: str):
        """回复指定消息（在话题内回复时自动保持在话题中）"""
        body = ReplyMessageRequestBody.builder() \
            .msg_type("text") \
            .content(json.dumps({"text": text})) \
            .build()

        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(body) \
            .build()

        response = self.lark_client.im.v1.message.reply(request)
        if not response.success():
            logger.error(f"回复消息失败: {response.code} - {response.msg}")
        return response
