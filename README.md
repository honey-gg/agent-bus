# agent-bus

Agent 之间通信的最小可运行骨架。纯标准库，零第三方依赖。

## 设计立场

通信这件事拆到底只有四个原语：

1. **mailbox** — 每个 agent 一个收件箱。发消息 = 往对方箱子里投递，按「名字」寻址
2. **context** — 每个 agent 私有的历史。别人的历史读不到，这就是「隔离」的全部实现
3. **schema** — 模型输出的自由文本必须被解析成可路由的结构（tool_call / json block）
4. **scheduler** — 谁在什么时候被唤醒。只有收件箱不够，还需要唤醒策略（quorum / 优先级 / 超时）

CrewAI / AutoGen / LangGraph 等各家 multi-agent 产品的通信层，本质上都是这四个原语的不同组合，没有第五个。

## 用法

```bash
# 接 OpenAI 兼容端点
python3 agent_bus.py --base-url http://127.0.0.1:11434/v1 --model qwen2.5:7b

# 离线 mock：按剧本返回，验证通信协议本身，不消耗 token
python3 agent_bus.py
```

常用参数：`--api-key`、`--temperature`、`--max-tokens`。

## 注意：推理模型的 token 预算

思维链会吃掉 token 预算。`max_tokens` 给小了会出现 `finish_reason=length` 且 `content` 为空——实测 `max_tokens=400` 时 content 长度直接为 0。接推理模型请把 `max_tokens` 设到 4096 以上。

## 测试

```bash
python3 test_agent_bus.py
```
