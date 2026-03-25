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
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from providers.base import AIProvider
from bot.session import SessionManager
from bot.agent_manager import AgentManager

logger = logging.getLogger(__name__)


def _build_approval_card(request_id: str, user_id: str, text: str) -> str:
    """构建审批卡片 JSON。"""
    # 截断过长的消息预览
    preview = text if len(text) <= 200 else text[:200] + "..."
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "🔒 Agent 执行审批"},
            "template": "orange",
        },
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**发送者：** <at id={user_id}></at>"},
            },
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**消息内容：**\n{preview}"},
            },
            {"tag": "hr"},
            {
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
            },
        ],
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
    ):
        self.providers = providers
        self.default_provider = default_provider
        self.session_manager = session_manager
        self.lark_client = lark_client
        self.agent_manager = agent_manager
        self.bot_open_id = bot_open_id
        self._feishu_cfg = feishu_cfg or {}
        self._admin_ids = admin_ids or []
        self._processed_msgs: dict[str, float] = {}
        # 待审批请求：request_id -> {message_id, text, images, user_id, timestamp}
        self._pending_requests: dict[str, dict] = {}

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

        # 忽略机器人自己发送的消息
        if self.bot_open_id and user_id == self.bot_open_id:
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

        # 群聊中需要 @机器人 才触发（私聊不需要）
        if chat_type == "group":
            is_bot_mentioned = any(
                getattr(getattr(m, "id", None), "open_id", None) == self.bot_open_id
                for m in mentions
            ) if self.bot_open_id else bool(mentions)
            if not is_bot_mentioned:
                return

        text = re.sub(r"@_user_\d+\s*", "", text).strip()
        if not text and not images:
            return

        # 只有图片没有文字也允许继续（agent_manager 会处理）
        if not text and images:
            text = ""

        # 话题内消息 → 路由到 Agent
        if thread_id:
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

        # 普通消息 → 管理员直接执行，非管理员需要审批
        if self._admin_ids and user_id not in self._admin_ids:
            self._send_approval_request(message_id, chat_id, text, images, user_id)
        else:
            threading.Thread(
                target=self._handle_new_agent_thread,
                args=(message_id, text, images),
                daemon=True,
            ).start()

    def _send_approval_request(self, message_id: str, chat_id: str, text: str,
                                images: list[dict], user_id: str):
        """发送审批卡片，等待管理员批准。"""
        request_id = uuid.uuid4().hex[:12]
        self._pending_requests[request_id] = {
            "message_id": message_id,
            "chat_id": chat_id,
            "text": text,
            "images": images,
            "user_id": user_id,
            "timestamp": time.time(),
        }
        # 清理超过 30 分钟的过期请求
        cutoff = time.time() - 1800
        self._pending_requests = {
            k: v for k, v in self._pending_requests.items()
            if v["timestamp"] > cutoff
        }

        card_json = _build_approval_card(request_id, user_id, text)
        body = ReplyMessageRequestBody.builder() \
            .msg_type("interactive") \
            .content(card_json) \
            .build()
        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(body) \
            .build()
        resp = self.lark_client.im.v1.message.reply(request)
        if not resp.success():
            logger.error("发送审批卡片失败: %s - %s", resp.code, resp.msg)

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
            # 启动 Agent
            threading.Thread(
                target=self._handle_new_agent_thread,
                args=(pending["message_id"], pending["text"], pending["images"]),
                daemon=True,
            ).start()
            return {
                "toast": {"type": "success", "content": "已批准，Agent 正在启动"},
                "card": self._build_result_card("✅ Agent 执行已批准", "green",
                    f"**审批人：** <at id={operator_id}></at>\n**状态：** 已批准，Agent 正在执行"),
            }
        else:
            logger.info("管理员 %s 拒绝了请求 %s", operator_id, request_id)
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
        def _run():
            # 先发提示
            resp = self._reply_to_message(message_id, "🤖 正在思考中...")
            hint_msg_id = None
            if resp and resp.success() and resp.data:
                hint_msg_id = getattr(resp.data, "message_id", None)

            try:
                reply = self.agent_manager.chat(thread_id, text, images=images)
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
