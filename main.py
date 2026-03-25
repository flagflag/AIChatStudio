import logging
import yaml

import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

from providers.base import ProviderFactory
# 导入以触发 register
import providers.claude_provider   # noqa: F401
import providers.gemini_provider   # noqa: F401
import providers.openai_provider   # noqa: F401

from bot.handler import MessageHandler
from bot.session import SessionManager
from bot.agent_manager import AgentManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_providers(config: dict) -> dict:
    """根据配置创建所有可用的 AI Provider 实例"""
    providers = {}
    for key, cfg in config.get("providers", {}).items():
        api_key = cfg.get("api_key", "")
        model = cfg.get("model", "")
        if not api_key or api_key.startswith("your-") or api_key.endswith("..."):
            logger.warning(f"跳过 {key}: API key 未配置")
            continue
        try:
            providers[key] = ProviderFactory.create(key, api_key, model)
            logger.info(f"已加载 provider: {providers[key].name}")
        except ValueError as e:
            logger.warning(f"跳过 {key}: {e}")
    return providers


def main():
    config = load_config()

    # 创建 AI providers
    providers = create_providers(config)
    if not providers:
        logger.error("没有可用的 AI provider，请检查 config.yaml 中的 API key 配置")
        return

    # 确定默认 provider
    default_provider = config.get("default_provider", "claude")
    if default_provider not in providers:
        default_provider = next(iter(providers))
        logger.warning(f"默认 provider 不可用，已切换为: {default_provider}")

    # 创建会话管理器
    session_cfg = config.get("session", {})
    session_manager = SessionManager(
        max_history=session_cfg.get("max_history", 20),
        timeout_minutes=session_cfg.get("timeout_minutes", 30),
    )

    # 创建飞书客户端
    feishu_cfg = config.get("feishu", {})
    lark_client = lark.Client.builder() \
        .app_id(feishu_cfg["app_id"]) \
        .app_secret(feishu_cfg["app_secret"]) \
        .log_level(lark.LogLevel.INFO) \
        .build()

    # 获取机器人自身信息
    bot_open_id = ""
    try:
        import httpx
        # 先获取 tenant_access_token
        token_resp = httpx.post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal", json={
            "app_id": feishu_cfg["app_id"],
            "app_secret": feishu_cfg["app_secret"],
        }).json()
        token = token_resp.get("tenant_access_token", "")
        if token:
            bot_resp = httpx.get("https://open.feishu.cn/open-apis/bot/v3/info",
                                 headers={"Authorization": f"Bearer {token}"}).json()
            bot_data = bot_resp.get("bot", {})
            bot_open_id = bot_data.get("open_id", "")
            logger.info(f"机器人信息: {bot_data.get('app_name', '')} ({bot_open_id})")
        else:
            logger.warning("获取 tenant_access_token 失败，群聊 @检测将降级为任意 @触发")
    except Exception:
        logger.exception("获取机器人信息异常，群聊 @检测将降级为任意 @触发")

    # 创建 Agent 管理器
    agent_cfg = config.get("agent", {})
    agent_manager = AgentManager(
        cwd=agent_cfg.get("cwd", "."),
        allowed_tools=agent_cfg.get("allowed_tools", ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]),
        timeout_minutes=agent_cfg.get("timeout_minutes", 120),
        max_agents=agent_cfg.get("max_agents", 3),
    )

    # 创建消息处理器
    handler = MessageHandler(
        providers=providers,
        default_provider=default_provider,
        session_manager=session_manager,
        lark_client=lark_client,
        agent_manager=agent_manager,
        bot_open_id=bot_open_id,
        feishu_cfg=feishu_cfg,
    )

    # 注册 Agent 清理通知回调
    agent_manager._on_cleanup = lambda msg_id, reason: handler._reply_to_message(msg_id, f"⚠️ {reason}")

    # 定义事件回调
    def on_message(data: P2ImMessageReceiveV1):
        try:
            msg = data.event.message
            event_data = {
                "message": {
                    "message_id": msg.message_id,
                    "chat_id": msg.chat_id,
                    "chat_type": getattr(msg, "chat_type", "") or "",
                    "message_type": msg.message_type,
                    "content": msg.content,
                    "thread_id": getattr(msg, "thread_id", None) or "",
                    "mentions": getattr(msg, "mentions", None) or [],
                },
                "sender": {
                    "sender_id": {
                        "open_id": data.event.sender.sender_id.open_id,
                    }
                },
            }
            handler.handle(event_data)
        except Exception:
            logger.exception("处理消息时出错")

    # 构建事件处理器
    event_handler = lark.EventDispatcherHandler.builder(
        "", ""  # verification token 和 encrypt key，WebSocket 模式下留空
    ).register_p2_im_message_receive_v1(on_message).build()

    # 使用 WebSocket 长连接模式启动（无需公网地址）
    cli = lark.ws.Client(
        feishu_cfg["app_id"],
        feishu_cfg["app_secret"],
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    logger.info("=" * 50)
    logger.info("飞书 AI 机器人已启动 (WebSocket 模式)")
    logger.info(f"默认模型: {providers[default_provider].name}")
    logger.info(f"已加载模型: {', '.join(p.name for p in providers.values())}")
    logger.info(f"Agent 工作目录: {agent_cfg.get('cwd', '.')}")
    logger.info("=" * 50)

    cli.start()


if __name__ == "__main__":
    main()
