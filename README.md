  使用步骤

  1. 安装依赖
  pip install -r requirements.txt

  2. 创建配置文件
  cp config.example.yaml config.yaml
  然后编辑 config.yaml，填入：
  - 飞书 App ID / Secret（在 https://open.feishu.cn/ 创建应用获取）
  - 各 AI 的 API Key（不用的 provider 保持占位符即可，会自动跳过）

  3. 飞书应用配置

  3.1 基础配置
  - 开启「机器人」能力

  3.2 事件订阅
  - im.message.receive_v1（接收消息）
  - card.action.trigger（卡片按钮回调，审批功能需要）

  3.3 权限（权限管理 → 搜索并开通）

  | 权限 | 说明 | 用途 |
  |------|------|------|
  | im:message | 获取与发送消息 | 接收/发送消息、拉取话题历史 |
  | im:message:send_as_bot | 以机器人身份发送消息 | 回复消息、创建话题 |
  | im:chat:readonly | 获取群信息 | 审批卡片中显示群聊名称 |
  | im:resource | 读取消息中的资源文件 | 下载用户发送的图片 |
  | contact:user.base:readonly | 获取用户基本信息 | 话题上下文中显示用户真实名字 |
  | docx:document:readonly | 读取文档内容 | 预取飞书文档供 Agent 阅读 |
  | wiki:wiki:readonly | 读取知识库 | 预取飞书知识库文档（wiki 链接） |
  | drive:drive:readonly | 读取云空间文件 | 下载文档中的图片 |

  3.4 发布应用
  - 以上配置完成后，需要创建应用版本并发布，权限才会生效

  4. 启动
  python main.py

  5. 群聊中使用
  - 直接 @机器人 发消息 → 使用默认模型（Claude）
  - @机器人 /gemini 你好 → 用 Gemini 回复
  - @机器人 /gpt 你好 → 用 OpenAI 回复
  - @机器人 /clear → 清除对话记忆
  - @机器人 /help → 查看帮助
  - @机器人 /myid → 查看自己的 open_id

  6. Agent 模式
  - 每条新消息会自动创建话题，话题内对话由独立 Claude Agent 处理
  - Agent 工作目录、允许的工具、超时时间等在 config.yaml 的 agent 部分配置
  - 支持图片识别（用户发送的图片会保存为临时文件供 Agent 读取）
  - 支持飞书文档预取（消息中包含飞书文档链接时自动拉取内容）

  7. 审批模式（可选）
  - 在 config.yaml 中配置 admin_ids，非管理员的消息需要审批才会执行
  - 审批卡片以私信形式发送给管理员
  - 可配置 approval_expire_minutes 设置批准有效期（默认 1440 分钟 = 24 小时）

  扩展新模型

  只需 3 步：
  1. 在 providers/ 下新建文件（如 deepseek_provider.py）
  2. 继承 AIProvider，实现 chat() 和 name
  3. 调用 ProviderFactory.register("deepseek", DeepSeekProvider) 注册，然后在 main.py 加一行 import 即可
