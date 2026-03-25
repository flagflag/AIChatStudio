import atexit
import asyncio
import base64
import logging
import os
import signal
import tempfile
import threading
import time
from collections.abc import Callable

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

logger = logging.getLogger(__name__)


class AgentSession:
    """单个 Agent 会话，绑定到一个飞书话题。"""

    def __init__(self, cwd: str, allowed_tools: list[str], system_prompt: str = "", temp_dir: str = ""):
        self._cwd = cwd
        self._allowed_tools = allowed_tools
        self._system_prompt = system_prompt
        self._temp_dir = temp_dir
        self._client: ClaudeSDKClient | None = None
        self._connected = False
        self._lock = asyncio.Lock()
        self.last_active = time.time()
        self.reply_message_id: str = ""  # 话题内可用于回复的 message_id

    async def _ensure_connected(self):
        if self._client is None:
            options = ClaudeAgentOptions(
                cwd=self._cwd,
                allowed_tools=self._allowed_tools,
                system_prompt=self._system_prompt or None,
                setting_sources=["project"],
                permission_mode="bypassPermissions",
            )
            self._client = ClaudeSDKClient(options=options)
            await self._client.connect()
            self._connected = True

    async def chat(self, message: str, images: list[dict] | None = None) -> str:
        async with self._lock:
            await self._ensure_connected()
            self.last_active = time.time()

            saved_files = []
            if images:
                prompt = self._save_images_and_build_prompt(message, images, saved_files, self._temp_dir)
            else:
                prompt = message

            try:
                await self._client.query(prompt)

                result_text = ""
                async for msg in self._client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                result_text += block.text
                    elif isinstance(msg, ResultMessage):
                        break

                return result_text
            finally:
                # 清理临时图片文件
                for f in saved_files:
                    try:
                        os.unlink(f)
                    except OSError:
                        pass

    @staticmethod
    def _save_images_and_build_prompt(text: str, images: list[dict], saved_files: list, temp_dir: str = "") -> str:
        """将图片保存为临时文件，构建包含文件路径的文本提示。"""
        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
        paths = []
        for img in images:
            ext = ext_map.get(img["media_type"], ".png")
            fd, path = tempfile.mkstemp(suffix=ext, prefix="feishu_img_", dir=temp_dir or None)
            try:
                os.write(fd, base64.b64decode(img["data"]))
            finally:
                os.close(fd)
            saved_files.append(path)
            paths.append(path)

        # 构建提示：让 Agent 先用 Read 工具读取图片再回复
        img_hint = "\n".join(p for p in paths)
        prompt = f"[系统] 用户通过飞书发送了 {len(paths)} 张图片，已保存到本地。你必须先使用 Read 工具读取以下图片文件，再回复用户。\n\n{img_hint}"
        if text:
            prompt += f"\n\n用户消息：{text}"
        return prompt

    async def close(self):
        if self._client and self._connected:
            try:
                await self._client.disconnect()
            except Exception:
                logger.exception("断开 agent 连接失败")
            self._client = None
            self._connected = False


class AgentManager:
    """管理所有 Agent 会话，按 thread_id 隔离。使用专用事件循环保持连接存活。"""

    def __init__(self, cwd: str, allowed_tools: list[str], timeout_minutes: int = 30, max_agents: int = 3,
                 system_prompt: str = "", temp_dir: str = "", on_cleanup: Callable[[str, str], None] | None = None):
        """
        Args:
            on_cleanup: 清理回调，参数为 (reply_message_id, reason)，用于向话题推送通知。
        """
        self._cwd = cwd
        self._allowed_tools = allowed_tools
        self._system_prompt = system_prompt
        self._temp_dir = temp_dir
        self._timeout = timeout_minutes * 60
        self._cleanup_interval = 5 * 60
        self._max_agents = max_agents
        self._on_cleanup = on_cleanup
        self._sessions: dict[str, AgentSession] = {}

        # 启动专用事件循环线程，所有 Agent 操作都在这个循环中执行
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        # 启动定时清理
        self._start_cleanup_timer()

        # 退出清理
        atexit.register(self.shutdown)
        # 链式信号处理：先清理 Agent，再调用原有处理器（不破坏飞书 WebSocket 的信号处理）
        self._original_handlers: dict[int, object] = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal_handler)

    def _run_loop(self):
        """在后台线程中运行专用事件循环。"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coroutine(self, coro, timeout: float | None = 30):
        """在专用事件循环中执行协程，阻塞等待结果。"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def chat(self, thread_key: str, message: str, images: list[dict] | None = None) -> str:
        """同步接口：在专用事件循环中执行 Agent 对话。"""
        return self._run_coroutine(self._async_chat(thread_key, message, images=images), timeout=None)

    def set_reply_message_id(self, thread_key: str, message_id: str):
        """记录话题内可用于回复的 message_id。"""
        if thread_key in self._sessions:
            self._sessions[thread_key].reply_message_id = message_id

    async def _async_chat(self, thread_key: str, message: str, images: list[dict] | None = None) -> str:
        if thread_key not in self._sessions:
            # 达到上限时关闭最久未活跃的 session
            while len(self._sessions) >= self._max_agents:
                oldest_key = min(self._sessions, key=lambda k: self._sessions[k].last_active)
                logger.info("Agent 数量已达上限 (%d)，关闭最久未活跃的 session: %s", self._max_agents, oldest_key)
                self._notify_cleanup(oldest_key, "Agent 已被回收（达到数量上限，优先回收最久未活跃的会话）")
                await self._sessions[oldest_key].close()
                del self._sessions[oldest_key]
            self._sessions[thread_key] = AgentSession(self._cwd, self._allowed_tools, self._system_prompt, self._temp_dir)
            logger.info("Created new agent session for thread: %s", thread_key)

        session = self._sessions[thread_key]
        return await session.chat(message, images=images)

    def bind_thread(self, temp_key: str, thread_id: str):
        """将临时 key 重新绑定到实际的 thread_id。"""
        if temp_key in self._sessions and temp_key != thread_id:
            self._sessions[thread_id] = self._sessions.pop(temp_key)
            logger.info("Bound agent session: %s -> %s", temp_key, thread_id)

    def _notify_cleanup(self, thread_key: str, reason: str):
        """通知话题 Agent 已被清理（在独立线程中执行，不阻塞事件循环）。"""
        if not self._on_cleanup:
            return
        session = self._sessions.get(thread_key)
        if session and session.reply_message_id:
            msg_id = session.reply_message_id
            threading.Thread(
                target=self._safe_notify,
                args=(msg_id, reason),
                daemon=True,
            ).start()

    def _safe_notify(self, msg_id: str, reason: str):
        try:
            self._on_cleanup(msg_id, reason)
        except Exception:
            logger.exception("发送清理通知失败")

    def _signal_handler(self, signum, frame):
        """链式信号处理：先清理 Agent，再调用原有处理器。"""
        logger.info("收到信号 %s，正在清理 Agent...", signal.Signals(signum).name)
        self.shutdown()
        # 调用原有处理器
        original = self._original_handlers.get(signum)
        if callable(original):
            original(signum, frame)
        elif original == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

    def _start_cleanup_timer(self):
        """启动后台定时清理。"""
        def _run():
            while True:
                time.sleep(self._cleanup_interval)
                try:
                    self._run_coroutine(self._async_cleanup())
                except TimeoutError:
                    logger.warning("定时清理超时")
                except Exception:
                    logger.exception("定时清理失败")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        logger.info("Agent 会话清理定时器已启动（间隔 %d 秒）", self._cleanup_interval)

    def shutdown(self):
        """关闭所有 Agent session 和事件循环。"""
        if not self._sessions:
            return
        logger.info("正在关闭 %d 个 Agent session...", len(self._sessions))
        try:
            # 通知所有话题
            for key in list(self._sessions):
                self._notify_cleanup(key, "Agent 已关闭（服务器停止）")
            self._run_coroutine(self._async_close_all())
        except Exception:
            logger.exception("关闭 Agent session 时出错")
        self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("所有 Agent session 已关闭")

    async def _async_close_all(self):
        for key, session in list(self._sessions.items()):
            await session.close()
            logger.info("已关闭 agent session: %s", key)
        self._sessions.clear()

    async def _async_cleanup(self):
        now = time.time()
        expired = [
            k for k, s in self._sessions.items()
            if now - s.last_active > self._timeout
        ]
        for key in expired:
            self._notify_cleanup(key, "Agent 已回收（超时未活跃）")
            await self._sessions[key].close()
            del self._sessions[key]
            logger.info("已清理过期 agent session: %s", key)
