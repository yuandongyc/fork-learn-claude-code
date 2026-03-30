# YD 学习计划：基于 OpenRouter 的 Agent 学习

## 目标

将 `agents/` 目录下的 Python 文件改编为使用 **OpenRouter** API（兼容 OpenAI 格式），手动逐个修改学习。

## 为什么用 OpenRouter

- 支持 Anthropic、OpenAI、DeepSeek、Qwen 等多种模型
- 可用支付宝/微信支付
- 兼容 OpenAI API 格式，只需修改 base_url 和 API Key

## 修改方案

### 核心改动

```python
# 改前 (Anthropic)
from anthropic import Anthropic
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
response = client.messages.create(model=MODEL, messages=messages, tools=TOOLS)

# 改后 (OpenRouter / OpenAI 兼容)
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/v1",
    api_key="your-api-key"
)
response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=messages,
    tools=TOOLS
)
```

### 需要适配的部分

1. **导入和初始化**
2. **API 调用**：`messages.create` → `chat.completions.create`
3. **响应格式**：
   - 工具调用：`block.type == "tool_use"` → `message.tool_calls`
   - 工具参数：`block.input` → `tool.function.arguments`
4. **工具定义格式**：OpenAI 的 function calling 格式略有不同

---

## 文件清单（共 13 个核心文件）

| 序号 | 文件 | 内容 | 难度 |
|------|------|------|------|
| 01 | `s01_agent_loop.py` | Agent Loop 基础 + bash 工具 | ⭐⭐⭐⭐⭐ |
| 02 | `s02_tool_use.py` | 添加工具（read/write/edit/glob/grep） | ⭐⭐⭐ |
| 03 | `s03_todo_write.py` | TodoWrite 工具 | ⭐⭐⭐ |
| 04 | `s04_subagent.py` | 子 Agent 机制 | ⭐⭐⭐⭐ |
| 05 | `s05_skill_loading.py` | Skill 按需加载 | ⭐⭐⭐ |
| 06 | `s06_context_compact.py` | 上下文压缩 | ⭐⭐⭐⭐ |
| 07 | `s07_task_system.py` | 任务系统 | ⭐⭐⭐⭐ |
| 08 | `s08_background_tasks.py` | 后台任务 | ⭐⭐⭐⭐ |
| 09 | `s09_agent_teams.py` | Agent 团队 | ⭐⭐⭐⭐⭐ |
| 10 | `s10_team_protocols.py` | 团队协议（审批/关闭） | ⭐⭐⭐⭐⭐ |
| 11 | `s11_autonomous_agents.py` | 自主 Agent | ⭐⭐⭐⭐⭐ |
| 12 | `s12_worktree_task_isolation.py` | Worktree 任务隔离 | ⭐⭐⭐⭐⭐ |
| 13 | `s_full.py` | 完整参考实现 | ⭐⭐⭐⭐⭐ |

---

## 学习顺序（推荐）

### 第一阶段：核心基础
1. **s01_agent_loop** - 理解 Agent Loop，这是最核心的改动
2. **s02_tool_use** - 添加各种工具

### 第二阶段：增强功能
3. **s03_todo_write** - 任务规划
4. **s04_subagent** - 子 Agent
5. **s05_skill_loading** - Skill 加载

### 第三阶段：高级功能
6. **s06_context_compact** - 上下文压缩
7. **s07_task_system** - 任务系统
8. **s08_background_tasks** - 后台任务

### 第四阶段：团队协作
9. **s09_agent_teams** - 多 Agent 协作
10. **s10_team_protocols** - 团队协议
11. **s11_autonomous_agents** - 自主 Agent
12. **s12_worktree_task_isolation** - 任务隔离

### 第五阶段：综合应用
13. **s_full** - 完整参考实现

---

## 准备工作

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install openai python-dotenv requests
```

### 2. 配置 OpenRouter

在 `.env` 文件中添加：

```env
OPENROUTER_API_KEY=your-openrouter-api-key
```

### 3. 选择模型（推荐免费）

#### 免费模型（推荐）
| 模型 | 上下文 | 输入 | 输出 |
|------|--------|------|------|
| **Step 3.5 Flash** | 256K | $0 | $0 |
| **NVIDIA Nemotron 3 Super** | 262K | $0 | $0 |

在 `.env` 中配置：
```env
MODEL_ID=stepfun/Step-3.5-Flash
# 或
MODEL_ID=nvidia/nemotron-3-super-4b-hf
```

#### 付费模型
```env
MODEL_ID=anthropic/claude-3.5-sonnet-20241022
# 或其他：
# MODEL_ID=openai/gpt-4o
# MODEL_ID=deepseek/deepseek-chat
# MODEL_ID=qwen/qwen-2.5-72b-instruct
```

### 4. 申请 OpenRouter API Key

访问 https://openrouter.ai/ 注册账号：
- 新用户有少量免费 credits
- 可用支付宝/微信充值
- 推荐先用免费模型测试

---

## 关键改动示例

### s01_agent_loop.py 改动要点

```python
# 1. 导入改用 OpenAI
from openai import OpenAI

# 2. 初始化
client = OpenAI(
    base_url="https://openrouter.ai/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
MODEL = os.getenv("MODEL_ID", "stepfun/Step-3.5-Flash")

# 3. API 调用
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": SYSTEM}] + messages,
    tools=TOOLS,
    max_tokens=8000
)

# 4. 工具调用提取
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        # 执行工具...
```

---

## 验证方法

每个文件修改完成后，运行测试：

```bash
python agents-yd/s01_agent_loop.py

chcp 65001  --防止中文乱码
venv\Scripts\python agents-yd\s01_agent_loop.py
```

如果看到 `s01 >>` 提示符，说明基本运行成功。

输入测试问题：
```
你好，列出当前目录的文件
```

---

## 参考资源

- OpenRouter 文档：https://openrouter.ai/docs
- OpenAI 兼容 API：https://openrouter.ai/docs/api-compatibility
- 原始项目：https://github.com/anomalyco/opencode (参考 s01-s12)

---

## 注意事项

1. **逐个文件修改**：不要一次性改所有文件，先改 s01 理解核心模式
2. **免费模型够用**：Step 3.5 Flash 完全免费，足够学习使用
3. **模型选择**：推荐先用免费的 Step 3.5 Flash 测试
4. **工具安全**：保留危险命令拦截（如 `rm -rf /`）

---

*计划制定完成，祝学习愉快！*
