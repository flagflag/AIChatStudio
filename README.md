  使用步骤

  1. 安装依赖
  cd C:\Workspace\AI-Agent-Server
  pip install -r requirements.txt

  2. 创建配置文件
  cp config.example.yaml config.yaml
  然后编辑 config.yaml，填入：
  - 飞书 App ID / Secret（在 https://open.feishu.cn/ 创建应用获取）
  - 各 AI 的 API Key（不用的 provider 保持占位符即可，会自动跳过）

  3. 飞书应用配置
  - 开启「机器人」能力
  - 添加权限：im:message（接收消息）、im:message:send_as_bot（发送消息）
  - 事件订阅中添加 im.message.receive_v1

  4. 启动
  python main.py

  5. 群聊中使用
  - 直接 @机器人 发消息 → 使用默认模型（Claude）
  - @机器人 /gemini 你好 → 用 Gemini 回复
  - @机器人 /gpt 你好 → 用 OpenAI 回复
  - @机器人 /clear → 清除对话记忆
  - @机器人 /help → 查看帮助

  扩展新模型

  只需 3 步：
  1. 在 providers/ 下新建文件（如 deepseek_provider.py）
  2. 继承 AIProvider，实现 chat() 和 name
  3. 调用 ProviderFactory.register("deepseek", DeepSeekProvider) 注册，然后在 main.py 加一行 import 即可
  