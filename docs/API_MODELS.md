# API 模型文档

## 一、动态获取的模型（Antigravity API）

通过 `fetch_available_models()` 从下游 API 动态获取，并额外添加 `claude-opus-4-5`。每个模型还会生成一个 `流式抗截断/` 前缀版本。

## 二、静态定义的基础模型

位于 `src/utils.py:67-72`：

```python
BASE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview"
]
```

每个基础模型会生成多种变体：
- 原始模型：`gemini-2.5-pro`
- 假流式：`假流式/gemini-2.5-pro`
- 流式抗截断：`流式抗截断/gemini-2.5-pro`
- Thinking 后缀：`-maxthinking`, `-nothinking`
- Search 后缀：`-search`

## 三、Anthropic 接口支持的模型

位于 `src/anthropic_converter.py:94-107`：

| 模型名 | 类型 |
|--------|------|
| `gemini-2.5-flash` | Gemini |
| `gemini-2.5-flash-thinking` | Gemini |
| `gemini-2.5-pro` | Gemini |
| `gemini-3-pro-low` | Gemini |
| `gemini-3-pro-high` | Gemini |
| `gemini-3-pro-image` | Gemini (图像) |
| `gemini-2.5-flash-lite` | Gemini |
| `gemini-2.5-flash-image` | Gemini (图像) |
| `claude-sonnet-4-5` | Claude |
| `claude-sonnet-4-5-thinking` | Claude |
| `claude-opus-4-5-thinking` | Claude |
| `gpt-oss-120b-medium` | GPT |

## 四、Claude 模型映射规则

| 输入模型 | 映射到 |
|----------|--------|
| `claude-opus-4-5` | `claude-opus-4-5-thinking` |
| `claude-sonnet-4-5` | `claude-sonnet-4-5` |
| `claude-haiku-4-5` | `gemini-2.5-flash` |
| `claude-3-5-sonnet-*` | `claude-sonnet-4-5` |
| `claude-3-haiku-*` | `gemini-2.5-flash` |

## 五、OpenAI 接口模型映射

位于 `src/antigravity_router.py:67-72`：

```python
mapping = {
    "claude-sonnet-4-5-thinking": "claude-sonnet-4-5",
    "claude-opus-4-5": "claude-opus-4-5-thinking",
    "gemini-2.5-flash-thinking": "gemini-2.5-flash",
}
```

## 六、Anthropic 接口结构

### 端点定义

antigravity router 提供两个 Anthropic 兼容端点（位于 `src/antigravity_anthropic_router.py`）：

| 端点 | 功能 |
|------|------|
| `POST /antigravity/v1/messages` | 主消息接口，支持流式/非流式 |
| `POST /antigravity/v1/messages/count_tokens` | Token 计数端点 |

### 请求结构

接收标准 Anthropic Messages API 格式：

```python
{
    "model": str,           # 必填，模型名
    "max_tokens": int,      # 必填
    "messages": list,       # 必填，消息列表
    "stream": bool,         # 可选，是否流式
    "thinking": dict,       # 可选，thinking 配置
    "system": str/list,     # 可选，系统指令
    "tools": list           # 可选，工具定义
}
```

### 响应结构

非流式响应遵循 Anthropic 标准格式：

```python
{
    "id": "msg_xxx",
    "type": "message",
    "role": "assistant",
    "model": str,
    "content": [{"type": "text", "text": "..."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": int, "output_tokens": int}
}
```

流式响应通过 `antigravity_sse_to_anthropic_sse()` 转换，发送以下事件：
- `message_start`
- `content_block_start`
- `content_block_delta`
- `message_delta`
- `message_stop`

### 核心转换流程

1. **模型映射** (`anthropic_converter.py:67-121`)：Claude 模型名 → 下游模型名
2. **请求转换** (`convert_anthropic_request_to_antigravity_components`)：将 Anthropic 请求转为下游组件（contents, system_instruction, tools, generation_config）
3. **响应转换**：下游响应 → Anthropic 格式
