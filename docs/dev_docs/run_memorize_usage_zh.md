# run_memorize.py 使用文档

## 概述

`run_memorize.py` 是一个群聊记忆存储脚本，用于读取符合 `GroupChatFormat` 格式的 JSON 文件，并通过 HTTP API 逐条存储到记忆系统中。

## 功能特性

- ✅ 读取并验证 GroupChatFormat 格式的 JSON 文件
- ✅ 支持 `assistant` 和 `companion` 两种场景
- ✅ 自动保存对话元数据（conversation-meta）
- ✅ 逐条调用 memorize 接口处理消息
- ✅ 提供格式验证模式
- ✅ 详细的日志输出

## 使用方法

### 1. 基本用法

通过 HTTP API 调用存储记忆（需要指定 scene）：

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories \
  --scene assistant
```

### 2. 使用 companion 场景

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories \
  --scene companion
```

### 3. 仅验证格式

验证输入文件格式是否正确，不执行存储（无需 API 地址）：

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --scene assistant \
  --validate-only
```

## 命令行参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `--input` | 是 | 输入的群聊 JSON 文件路径（GroupChatFormat 格式） |
| `--scene` | 是 | 记忆提取场景，仅支持 `assistant` 或 `companion` |
| `--api-url` | 否* | memorize API 地址（非验证模式必需） |
| `--validate-only` | 否 | 仅验证输入文件格式，不执行存储 |

*注：使用 `--validate-only` 时不需要提供 `--api-url`，其他情况下必需。

## 输入文件格式

输入文件必须符合 `GroupChatFormat` 规范，详见 `data_format/group_chat/group_chat_format.py`。

### 格式示例

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "name": "智能销售助手项目组",
    "description": "智能销售助手项目的开发讨论群",
    "group_id": "group_sales_ai_2025",
    "created_at": "2025-02-01T01:00:00Z",
    "default_timezone": "UTC",
    "user_details": {
      "user_101": {
        "full_name": "Alex",
        "role": "技术负责人"
      },
      "user_102": {
        "full_name": "Betty",
        "role": "产品经理"
      }
    },
    "tags": ["AI", "销售", "项目开发"]
  },
  "conversation_list": [
    {
      "message_id": "msg_001",
      "create_time": "2025-02-01T02:00:00Z",
      "sender": "user_101",
      "sender_name": "Alex",
      "type": "text",
      "content": "大家早，今天讨论一下项目进度",
      "refer_list": []
    }
  ]
}
```

## 处理流程

脚本执行以下步骤：

1. **格式验证**
   - 读取输入 JSON 文件
   - 验证是否符合 GroupChatFormat 规范
   - 输出数据统计信息

2. **保存对话元数据**
   - 调用 `conversation-meta` 接口
   - 保存 scene、群组信息、用户详情等元数据
   - API 地址：`{base_url}/api/v1/conversation-meta`

3. **逐条处理消息**
   - 按顺序逐条调用 `memorize` 接口
   - 每条消息包含：message_id, create_time, sender, content 等
   - 自动添加 group_id, group_name, scene 信息
   - API 地址：`{api_url}` (即 `--api-url` 参数指定的地址)

4. **输出结果**
   - 显示成功处理的消息数量
   - 显示总共保存的记忆条数

## 输出示例

### 成功输出

```
🚀 群聊记忆存储脚本
======================================================================
📄 输入文件: /path/to/group_chat.json
🔍 验证模式: 否
🌐 API地址: http://localhost:1995/api/v1/memories
======================================================================
======================================================================
验证输入文件格式
======================================================================
正在读取文件: /path/to/group_chat.json
正在验证 GroupChatFormat 格式...
✓ 格式验证通过！

=== 数据统计 ===
格式版本: 1.0.0
群聊名称: 智能销售助手项目组
群聊ID: group_sales_ai_2025
用户数量: 5
消息数量: 8
时间范围: 2025-02-01T02:00:00Z ~ 2025-02-01T02:05:00Z

======================================================================
读取群聊数据
======================================================================
正在读取文件: /path/to/group_chat.json
使用简单直接的单条消息格式，逐条处理

======================================================================
开始逐条调用 memorize API
======================================================================
群组名称: 智能销售助手项目组
群组ID: group_sales_ai_2025
消息数量: 8
API地址: http://localhost:1995/api/v1/memories

--- 保存对话元数据 (conversation-meta) ---
正在保存对话元数据到: http://localhost:1995/api/v1/conversation-meta
Scene: assistant, Group ID: group_sales_ai_2025
  ✓ 对话元数据保存成功
  Scene: assistant

--- 处理第 1/8 条消息 ---
  ✓ 成功保存 1 条记忆

--- 处理第 2/8 条消息 ---
  ⏳ 等待情景边界

--- 处理第 3/8 条消息 ---
  ✓ 成功保存 2 条记忆

--- 处理第 4/8 条消息 ---
  ⏳ 等待情景边界

--- 处理第 5/8 条消息 ---
  ⏳ 等待情景边界

--- 处理第 6/8 条消息 ---
  ✓ 成功保存 1 条记忆

--- 处理第 7/8 条消息 ---
  ⏳ 等待情景边界

--- 处理第 8/8 条消息 ---
  ✓ 成功保存 2 条记忆

======================================================================
处理完成
======================================================================
✓ 成功处理: 8/8 条消息
✓ 共保存: 6 条记忆

======================================================================
✓ 处理完成！
======================================================================
```

## 错误处理

### 文件不存在

```
错误: 输入文件不存在: /path/to/file.json
```

### 格式验证失败

```
✗ 格式验证失败！
请确保输入文件符合 GroupChatFormat 规范
```

### JSON 解析错误

```
✗ JSON 解析失败: Expecting value: line 1 column 1 (char 0)
```

## 开发说明

### 核心依赖

- `infra_layer.adapters.input.api.mapper.group_chat_converter`: 格式验证
- `httpx`: HTTP 客户端（异步请求）
- `core.observation.logger`: 日志工具

### API 端点

脚本会调用两个 API 端点：

1. **conversation-meta**: 保存对话元数据
   - 路径：`{base_url}/api/v1/conversation-meta`
   - 方法：POST
   - 数据：包含 scene, group_id, user_details 等元数据

2. **memorize**: 存储单条消息记忆
   - 路径：`{api_url}` (通过 `--api-url` 参数指定)
   - 方法：POST
   - 数据：包含 message_id, sender, content, scene 等

### 扩展建议

1. **批量处理**: 支持处理目录下的多个文件
2. **进度显示**: 添加进度条显示处理状态
3. **错误重试**: 添加失败重试机制
4. **并发处理**: 支持批量并发调用 API（注意保持消息顺序）
5. **结果导出**: 将存储结果导出为 JSON 文件

## 常见问题

### Q1: 为什么推荐使用 bootstrap.py 启动？

A: `bootstrap.py` 会自动处理：
- Python 路径设置
- 环境变量加载
- 依赖注入容器初始化
- Mock 模式支持

这样可以确保脚本在完整的应用上下文中运行。

### Q2: assistant 和 companion 场景有什么区别？

A: 
- **assistant**: 助理场景，适用于 AI 助理与用户的对话
- **companion**: 陪伴场景，适用于 AI 伙伴的互动对话

不同场景会影响记忆的提取策略和存储方式，需要根据实际应用场景选择。

### Q3: 为什么消息处理后显示"等待情景边界"？

A: 记忆系统使用"情景边界"（Episode Boundary）来判断何时形成完整的记忆片段。
- 并非每条消息都会立即生成记忆
- 系统会等待一个完整的对话情景结束后才提取记忆
- 这是正常的处理行为，不代表失败

### Q4: 可以不提供 API 地址吗？

A: 不可以。当前版本只支持通过 HTTP API 调用，必须提供 `--api-url` 参数（除非使用 `--validate-only` 仅验证格式）。

### Q5: API 调用失败了怎么办？

A: 检查以下几点：
1. 确保记忆服务正在运行
2. 确认 API 地址正确（包括端口号）
3. 查看服务端日志了解详细错误信息
4. 确认输入数据格式正确

## 参考资料

- [GroupChatFormat 格式定义](../../data_format/group_chat/group_chat_format.py)
- [Agentic V3 API 文档](../api_docs/agentic_v3_api_zh.md)
- [Bootstrap 使用文档](./bootstrap_usage.md)

