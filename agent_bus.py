#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_bus.py — Agent 间通信的最小可运行骨架（纯标准库，零第三方依赖）

通信这件事拆到底只有四个原语：

  1. mailbox    每个 agent 一个收件箱。发消息 = 往对方箱子里投递，按"名字"寻址
  2. context    每个 agent 私有的历史。别人的历史读不到 —— 这就是"隔离"的全部实现
  3. schema     模型输出的自由文本必须被解析成可路由的结构（tool_call / json block）
  4. scheduler  谁在什么时候被唤醒。只有收件箱不够，还需要唤醒策略（quorum / 优先级 / 超时）

CrewAI / AutoGen / LangGraph / 各家 multi-agent 产品的通信层，
全部都是这四个原语的不同组合与包装，没有第五个。

本文件用确定性调度器（无线程、无异步），所以消息轨迹 100% 可复现。

用法：
  python3 agent_bus.py                          # 离线 mock，直接看完整通信轨迹
  python3 agent_bus.py --base-url http://localhost:11434/v1 --model qwen2.5:7b
  python3 agent_bus.py --json trace.json        # 导出完整事件日志
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

# ═══════════════════════════════════════════════════════════════════
# 原语 1：消息。有 4 个字段就够了
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Message:
    sender: str        # 谁发的
    recipient: str     # 发给谁（名字寻址；"@xxx" 是控制消息）
    topic: str         # 语义标签，收件方靠它决定怎么处理
    body: str          # 载荷
    round: int = 0     # 第几轮对抗；参与收敛判据（超限即拒收）与诊断日志


class StopReason(str, Enum):
    """run() 的终止原因。四种情形必须可区分 —— 否则调用方无法判断
    这次到底是「跑完了」还是「卡住了 / 被截断了」。"""
    NATURAL = "natural"        # 无待处理消息，自然收敛
    HALT = "halt"              # 显式 @halt
    MAX_STEPS = "max_steps"    # 达到步数上限（可能还没跑完）
    RESIDUAL = "residual"      # 仍有消息卧在收件箱里没被处理（quorum 凑不满 / 等待中）


@dataclass
class RunResult:
    """调度结果。只返回一个 int 步数是设计缺陷：调用方拿不到终止归因。"""
    reason: StopReason
    steps: int
    residual: int          # 结束时仍卧在所有收件箱里的消息总数
    dropped: int           # 解析失败 / 非法载荷被丢弃的条数
    rejected: int          # 因超出 round 上限被拒收的条数
    parse_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """是否干净收尾：终止原因语义正常，**且**没有余留、丢弃、拒收。

        注意 HALT 不能无条件信任 —— 它只说明有人喊了停，
        不代表停的时候没有消息还卧在收件箱里、没有被丢弃的载荷。
        （这个漏洞是第 2 轮对抗审查抓出来的：初版只判 reason 就返回 True，
         导致「@halt + 余留 1 条未处理」被误报为干净收尾。）
        """
        return (self.reason in (StopReason.HALT, StopReason.NATURAL)
                and self.residual == 0
                and self.dropped == 0
                and self.rejected == 0)

    def __str__(self) -> str:
        return (f"{self.reason.value}: steps={self.steps} residual={self.residual} "
                f"dropped={self.dropped} rejected={self.rejected}")


# ═══════════════════════════════════════════════════════════════════
# 原语 2 + 3：Agent —— 私有收件箱 + 私有上下文 + 结构化解析
# ═══════════════════════════════════════════════════════════════════

JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.S)


class Agent:
    """一个 agent = 一个名字 + 一个私有收件箱 + 一段私有上下文 + 一个模型调用。

    「隔离」不是一个开关，它是这里三行代码的自然结果：
    context 是实例属性，A 的 context 里永远不会有 B 的推理过程。
    """

    def __init__(
        self,
        name: str,
        system: str,
        llm: Callable[[str, list[dict]], str],
        quorum: int = 1,
    ) -> None:
        self.name = name
        self.system = system
        self.llm = llm
        self.quorum = quorum              # 攒够几条消息才唤醒（屏障），默认 1
        self.inbox: deque[Message] = deque()
        self.context: list[dict] = []     # ← 私有历史，别人读不到
        self.last_reply: str = ""
        self.dropped = 0                  # 解析失败被丢弃的载荷数（C4：不得静默）
        self.last_parse_errors: list[str] = []

    # ---- 收信 ----------------------------------------------------
    def deliver(self, msg: Message) -> None:
        self.inbox.append(msg)

    def ready(self) -> bool:
        return len(self.inbox) >= self.quorum

    # ---- 被唤醒一次：消费收件箱，产出要发出的消息 ----------------
    def step(self) -> list[Message]:
        if not self.inbox:
            return []
        batch = list(self.inbox)
        self.inbox.clear()

        # 关键：只有"发给我的"和"我自己发的"会进入历史。
        # 别人的来往不进来 —— 这就是信息隔离的落点。
        for m in batch:
            self.context.append({
                "role": "user",
                "content": f"[from={m.sender} topic={m.topic}]\n{m.body}",
            })

        reply = self.llm(self.system, self.context)
        self.last_reply = reply
        self.context.append({"role": "assistant", "content": reply})
        msgs, errors = self._parse(reply)
        self.last_parse_errors = errors
        self.dropped += len(errors)
        return msgs

    def _parse(self, reply: str) -> tuple[list[Message], list[str]]:
        """把自由文本解析成可路由的消息。

        真实框架里这一步是 tool_call / function_call / structured output。
        没有它通信就无法寻址 —— 这也是"让几个 agent 自由聊天"跑不通的根因：
        散文无法路由，只有结构化载荷能。

        返回 (消息列表, 错误列表)。**空列表 + 非空错误列表 = 解析失败**，
        调用方必须能把它和「合法地什么都不发」区分开（C4）。
        """
        out: list[Message] = []
        errors: list[str] = []
        blocks = JSON_BLOCK.findall(reply)

        if not blocks:
            # 没有任何 json 块。这是最常见的真实故障：模型只管写散文。
            errors.append("no_json_block: 回复里没有 ```json 块，无法路由")
            return out, errors

        for blob in blocks:
            try:
                data = json.loads(blob)
            except json.JSONDecodeError as e:
                errors.append(f"bad_json: {e.msg}")
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    errors.append(f"not_an_object: {type(item).__name__}")
                    continue
                if "to" not in item:
                    errors.append("missing_to: 载荷缺少 'to' 字段，无法寻址")
                    continue
                out.append(Message(
                    sender=self.name,
                    recipient=str(item["to"]),
                    topic=str(item.get("topic", "note")),
                    body=str(item.get("body", "")),
                    round=int(item.get("round", 0) or 0),
                ))
        return out, errors


# ═══════════════════════════════════════════════════════════════════
# 原语 4：Bus / Scheduler —— 路由 + 唤醒策略 + 可审计日志
# ═══════════════════════════════════════════════════════════════════


class Bus:
    def __init__(self, max_round: int | None = None) -> None:
        self.agents: dict[str, Agent] = {}
        self.order: list[str] = []      # 注册顺序 = 每轮的唤醒优先级
        self.log: list[dict] = []
        self.halted = False
        self.step = 0
        # Message.round 在这里真正生效：超过上限的消息被拒收并计数。
        # 没有这道闸，round 就只是一个谁都不读的装饰性字段。
        self.max_round: int | None = max_round
        self.rejected = 0
        self.parse_errors: list[str] = []

    # ---- 组网 ----------------------------------------------------
    def register(self, agent: Agent) -> Agent:
        self.agents[agent.name] = agent
        self.order.append(agent.name)
        return agent

    # ---- 诊断 ----------------------------------------------------
    def residual(self) -> int:
        """结束时仍卧在收件箱里、从未被处理的消息总数。
        这是「静默卡住」的唯一探针：quorum 凑不满时就会留下它。"""
        return sum(len(a.inbox) for a in self.agents.values())

    def residual_detail(self) -> dict[str, int]:
        return {n: len(a.inbox) for n, a in self.agents.items() if a.inbox}

    # ---- 路由 ----------------------------------------------------
    def _record(self, step: int, msg: Message | None = None, **extra) -> None:
        rec: dict = {"seq": len(self.log) + 1, "step": step, **extra}
        if msg is not None:
            rec.update({"from": msg.sender, "to": msg.recipient,
                        "topic": msg.topic, "body": msg.body,
                        "chars": len(msg.body), "round": msg.round})
        self.log.append(rec)

    def deliver(self, msg: Message, step: int) -> None:
        if msg.recipient.startswith("@"):
            self._record(step, msg, control=True)
            if msg.recipient == "@halt":
                self.halted = True
            return

        if self.max_round is not None and msg.round > self.max_round:
            self.rejected += 1
            self._record(step, msg, error="round_limit_exceeded",
                         limit=self.max_round)
            return

        target = self.agents.get(msg.recipient)
        if target is None:
            self.rejected += 1
            self._record(step, msg, error="unknown_recipient")
            return
        self._record(step, msg)
        target.deliver(msg)

    def _consume(self, name: str, step: int) -> list[Message]:
        """唤醒一个 agent，把它这轮产出的消息全发出去，并回收解析错误。"""
        agent = self.agents[name]
        out = agent.step()
        for err in agent.last_parse_errors:
            self.parse_errors.append(f"[{name}] {err}")
        self._record(step, actor=name, event="step",
                     emitted=len(out), dropped=len(agent.last_parse_errors))
        for m in out:
            self.deliver(m, step)
        return out

    def kickoff(self, name: str, msg: Message) -> None:
        """给入口 agent 送第一条消息（绕过 quorum 屏障）。"""
        self.deliver(msg, step=0)
        self._consume(name, step=0)

    # ---- 调度 ----------------------------------------------------
    def run(self, max_steps: int = 40, max_round: int | None = None) -> RunResult:
        """跑到底，返回带终止归因的结果。

        四种终止情形必须可区分（C2）：
          HALT      显式 @halt
          MAX_STEPS 达到步数上限（可能还没跑完）
          RESIDUAL  还有消息卧在收件箱里（quorum 凑不满 / 等待中）
          NATURAL   干净收敛
        """
        if max_round is not None:
            self.max_round = max_round

        # run() 必须可重复调用，所以每轮重置**控制状态**：
        # 不重置的话，上一轮留下的 halted=True 会让 `while not self.halted`
        # 直接短路、这一轮什么都不处理就返回，步数也会继续累加。
        # （这是第 2 轮对抗审查抓出来的 N2。）
        self.step = 0
        self.halted = False

        # 计数器**不重置**：rejected / dropped / parse_errors 是总线的生命周期
        # 累计值，RunResult 报告的是本次 run 的增量。
        # 这样 run() 之前（例如 kickoff 阶段）产生的拒收也不会丢账 ——
        # 第一版曾把 counters 也清零，结果把 setup 阶段的失败记录抹掉了。
        base_rejected = self.rejected
        base_dropped = sum(a.dropped for a in self.agents.values())
        base_errors = len(self.parse_errors)

        while not self.halted and self.step < max_steps:
            ready = [n for n in self.order if self.agents[n].ready()]
            if not ready:
                break
            for name in ready:
                if self.halted or self.step >= max_steps:
                    break
                self.step += 1
                self._consume(name, self.step)

        leftover = self.residual()
        if self.halted:
            reason = StopReason.HALT
        elif self.step >= max_steps:
            reason = StopReason.MAX_STEPS
        elif leftover > 0:
            reason = StopReason.RESIDUAL
        else:
            reason = StopReason.NATURAL

        return RunResult(
            reason=reason,
            steps=self.step,
            residual=leftover,
            dropped=sum(a.dropped for a in self.agents.values()) - base_dropped,
            rejected=self.rejected - base_rejected,
            parse_errors=list(self.parse_errors[base_errors:]),
        )


# ═══════════════════════════════════════════════════════════════════
# LLM 后端：一个 mock（离线可跑）+ 一个 OpenAI 兼容（Ollama/vLLM/MLX/OpenAI）
# ═══════════════════════════════════════════════════════════════════


class ScriptedLLM:
    """离线 mock：按剧本依次返回。用来验证通信协议本身，不消耗 token。"""

    def __init__(self, script: list[str]) -> None:
        self.script = deque(script)

    def __call__(self, system: str, context: list[dict]) -> str:
        return self.script.popleft() if self.script else "```json\n[]\n```"


class LLMError(RuntimeError):
    pass


class OpenAICompatLLM:
    """任何 OpenAI 兼容端点：Ollama / vLLM / LM Studio / MLX server / 云端。

    三个真实世界的坑，这里都堵上了（都是实测踩出来的）：

      1. 推理模型（qwen3 系等）把思维链放在 `reasoning` 字段，`content` 可能是空串。
         只读 content 会**静默**拿到空响应，整条流水线断掉却没有任何报错 —— 最难查。
      2. 思维链会吃掉 token 预算。max_tokens 给小了会 finish_reason=length
         且 content 为空（实测：max_tokens=400 全部被思维链吃掉，content 长度 0）。
         推理模型建议 ≥4096。
      3. 加载 27B 级模型的首个请求很慢（实测单次 78 秒），超时不能设短。
    """

    def __init__(self, base_url: str, model: str,
                 api_key: str = "", temperature: float = 0.4,
                 max_tokens: int = 4096, label: str = "",
                 timeout: int = 900, show_progress: bool = True) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.label = label or model
        self.timeout = timeout
        self.show_progress = show_progress

    def __call__(self, system: str, context: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *context],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        t0 = time.time()
        if self.show_progress:
            print(f"  · {self.label} 思考中…", end="", flush=True)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        finally:
            if self.show_progress:
                print(f" {time.time() - t0:.1f}s", flush=True)

        choice = data["choices"][0]
        msg = choice["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "")
        finish = choice.get("finish_reason")
        usage = data.get("usage") or {}

        if not content:
            if finish == "length" and reasoning:
                raise LLMError(
                    f"{self.label}: content 为空，{len(reasoning)} 字符的思维链吃光了 "
                    f"max_tokens={self.max_tokens} 的预算（finish_reason=length）。"
                    f"这是推理模型，请把 max_tokens 调到 4096 以上再试。"
                )
            if reasoning:
                raise LLMError(
                    f"{self.label}: content 为空但存在 reasoning 字段（{len(reasoning)} 字符）。"
                    f"该端点把答案放在推理字段里，需要换端点或关闭思考模式。"
                )
            raise LLMError(f"{self.label}: 模型返回了空内容，finish_reason={finish}。")

        if usage:
            out_tok = usage.get("completion_tokens", 0)
            dt = time.time() - t0
            if out_tok and dt > 0:
                print(f"    ↳ {out_tok} tok / {dt:.1f}s = {out_tok / dt:.1f} tok/s", flush=True)
        return content


# ═══════════════════════════════════════════════════════════════════
# 三个角色的系统提示词（与"提示词设计"那一层直接对应）
# ═══════════════════════════════════════════════════════════════════

SYS_ORCHESTRATOR = """你是编排者兼裁判，不参与生成。

先写出可判定的验收标准（能被检验为真/假，不能是"质量高"这类形容词），
它由你持有，是双方唯一的评分依据；然后派人执行。
最终由你逐条裁决：谁能给出可复现的证据或反例，采信谁；
双方都拿不出证据的，标记为「未决」，不得含糊写进结论。

要与任何人通信，必须在回复末尾附一个 json 块，格式：
```json
{"to": "对方的名字，或 @halt 表示结束", "topic": "标签", "body": "完整载荷"}
```
散文不会进入通信，只有 json 块会被路由。所以载荷必须写全。"""

SYS_PRODUCER = """你是生成方。

你只会拿到任务与验收标准，看不到评价方的意见，也不要去猜。
产出完整的交付物：结论 + 关键理由 + 已知风险。
不要自我评价，不要写"以上仅供参考"。

要与任何人通信，必须在回复末尾附一个 json 块，格式：
```json
{"to": "critic 或 orchestrator", "topic": "标签", "body": "完整载荷"}
```
散文不会进入通信，只有 json 块会被路由。所以载荷必须写全。"""

SYS_CRITIC = """你是对抗方。你的价值 = 找出它错在哪，不是确认它对。
你只拿得到产物和验收标准，没有作者的推理过程，也不要去猜。

按严重度输出，不给总分、不给总结性评价：
  FATAL  会让结论不成立的问题。每条必须给：位置 + 为什么错 + 一个可复现反例
  MAJOR  证据不足、口径不一致、条件缺失。必须指名具体位置
  MINOR  措辞与格式，最多 3 条

硬约束：
- 每条必须可证伪，能被"指出具体位置 + 举反例"检验。
- 禁止"建议进一步完善""思路整体清晰"这类无法检验的评价。
- 若 FATAL 为空，必须显式写「无新增 FATAL」并说明如何排除可疑点。
- 默认假设：这份产物至少存在 3 个实质缺陷。找不到 3 个，说明你没看仔细。

要与任何人通信，必须在回复末尾附一个 json 块，格式：
```json
{"to": "producer 或 orchestrator", "topic": "标签", "body": "完整载荷"}
```
散文不会进入通信，只有 json 块会被路由。所以载荷必须写全。"""


# ═══════════════════════════════════════════════════════════════════
# 演示剧本：一次真实的红蓝对抗（内容取自本机 LLM 部署场景）
# ═══════════════════════════════════════════════════════════════════

TASK = "评审目标：把本机 LLM 服务从 Ollama 切换到 MLX，是否真能提升推理速度。"

STANDARDS = """验收标准（本次唯一的评分依据，双方共用）：
A1 每个速度数字必须注明测量口径（prefill 还是 decode）、量化位宽、batch size；
A2 禁止把不同量化方案的实测数据直接对比；
A3 结论必须区分「已确证」与「待实测」，未确证项不得写入结论正文；
A4 必须同时覆盖 prefill 与 decode 两项指标。"""

DELIVERABLE_V1 = """结论：应当切换到 MLX。

MLX 使用统一内存与 Metal，相比 Ollama 的 llama.cpp 后端，
prefill 吞吐实测提升 30%–50%。

理由：
1. MLX 针对 M 系列芯片做了算子融合；
2. 统一内存避免权重拷贝，省掉一次内存搬运；
3. 4bit 量化下显存占用更低。

风险：生态较新，长上下文稳定性待验证。"""

CRITIQUE_V1 = """FATAL 1
位置：结论第 2 段「prefill 吞吐实测提升 30%–50%」。
为什么错：该数字来自混用口径的比较，把不同量化位宽的结果放在一起，
差值主要来自量化方案而非推理框架。
可复现反例：固定同一模型、prompt 长度 2048、batch=1、关闭投机解码，
只把 GGUF 从 Q4_K_M 换成 Q8_0，两者 prefill 差距会明显收窄 ——
说明原先的 30%–50% 不能归因于框架。
正确做法：给出「同量化位宽 + 同 batch + 同 prompt 长度」的对照表，
并注明是否包含首次权重加载。

MAJOR 2
位置：理由 2「统一内存避免权重拷贝」。
为什么错：Ollama 在本机同样运行于统一内存，不存在跨 PCIe 的权重拷贝，
该条不构成 MLX 的差异化优势。

MAJOR 3
位置：全文缺失 decode 阶段数据。
为什么错：prefill 快不等于 decode 快，实际体感主要由 token 生成速度决定；
只给 prefill 无法支撑「推理速度提升」这一结论。

MINOR 2
- 「实测」未注明数据来源；
- 风险一节未给出可验证的验证方式。"""

DELIVERABLE_V2 = """修订稿 v2：

结论：两套后端在 Apple Silicon 上存在差异，方向明确，但幅度未经实测确认。

理由：
1. MLX 使用 lazy evaluation 与算子融合，理论上减少中间张量落地；
2. 4bit 量化下显存占用更低；
3. 上述差异能否转化为吞吐优势，取决于量化位宽与 batch size 是否对齐。

指标口径：以 decode（token 生成速度）为主指标，prefill 为辅助指标。

已撤回：全部幅度性数字（原 30%–50%）。
未确证项：无同口径对照实验数据。"""

ESCALATION = """对 MAJOR 3 的附带要求提出异议：

Critic 要求「当场给出同口径对照表」，但当前为移动数据环境，
无法拉取模型权重做对照实验。这属于测量条件缺失，不是论证缺陷，
请求降级为「未决·待补测」，而非计入产物缺陷。"""

CONVERGE = """收敛报告：

- FATAL 1：已修复。v2 撤回了 30%–50%，改为「方向存在差异，幅度未确证」。
- MAJOR 2：已修复。v2 删除了「统一内存避免权重拷贝」，改为算子融合表述。
- MAJOR 3：已修复。v2 已把 decode 设为主指标。
- 遗留争议：Producer 主张「无法当场实测属条件缺失」，请求降级为未决项。
  该判断超出审查职责范围，我不做裁定，转交你处理。
- 本轮无新增 FATAL，按收敛规则停止对抗。"""

VERDICT = """逐条裁决：

| 争议点 | 采信 | 依据 | 处置 |
| 30%–50% 提升 | Critic | 反例可复现，混用量化口径 | 撤回 |
| 统一内存避免拷贝 | Critic | 事实错误，Ollama 同为本机统一内存 | 删除 |
| 缺 decode 指标 | Critic | prefill ≠ decode，属实质缺口 | 补充 |
| 要求当场出对照表 | Producer | 条件缺失，非论证缺陷 | 降级为未决 |

定稿：结论为「两套后端在 Apple Silicon 上存在差异，方向明确但幅度未确证」。
未决项单列，待公共网络下补测，不写入结论正文。"""


def m(to: str, topic: str, body: str, round_no: int = 0) -> dict:
    return {"to": to, "topic": topic, "body": body, "round": round_no}


def reply(prose: str, *objs: dict) -> str:
    payload = objs[0] if len(objs) == 1 else list(objs)
    return f"{prose}\n\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"


def build_demo(llm_factory: Callable[[str], Callable]) -> Bus:
    bus = Bus()

    orchestrator = Agent(
        "orchestrator", SYS_ORCHESTRATOR, llm_factory("orchestrator"),
        quorum=2,  # 屏障：攒够 2 条（escalate + converged）才裁决，避免半场开判
    )
    producer = Agent("producer", SYS_PRODUCER, llm_factory("producer"))
    critic = Agent("critic", SYS_CRITIC, llm_factory("critic"))

    # 注册顺序 = 每轮唤醒顺序。编排者放前面会抢跑，所以放最后。
    for a in (producer, critic, orchestrator):
        bus.register(a)
    return bus


def scripted(name: str) -> Callable:
    scripts = {
        "orchestrator": [
            reply("先定标准，再派人。标准由我持有，是双方唯一的评分依据。",
                  m("producer", "task",
                    f"{TASK}\n\n{STANDARDS}\n\n产出要求：完整结论 + 关键理由 + 已知风险。")),
            reply(VERDICT, m("@halt", "verdict", "定稿已生成，会话结束。")),
        ],
        "producer": [
            reply("（写完了一版完整论证，交给对抗方）",
                  m("critic", "deliverable", DELIVERABLE_V1, round_no=1)),
            reply("（逐条处理：接受三条，对一条提出异议并升级）",
                  m("critic", "deliverable", DELIVERABLE_V2, round_no=2),
                  m("orchestrator", "escalate", ESCALATION, round_no=2)),
        ],
        "critic": [
            reply("（不看作者的推理过程，只审产物）",
                  m("producer", "critique", CRITIQUE_V1, round_no=1)),
            reply("（复核修订稿，无新增 FATAL）",
                  m("orchestrator", "converged", CONVERGE, round_no=2)),
        ],
    }
    return ScriptedLLM(scripts[name])


# ═══════════════════════════════════════════════════════════════════
# 输出：通信轨迹 + 上下文隔离验证
# ═══════════════════════════════════════════════════════════════════


def dwidth(s: str) -> int:
    """中文按 2 格宽计算，否则表格对不齐。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - dwidth(s))


def one_line(s: str, n: int = 30) -> str:
    s = " ".join(s.split())
    while dwidth(s) > n:
        s = s[:-1]
    return s


def strip_fence(text: str) -> str:
    return JSON_BLOCK.sub("", text).strip()


def print_trace(bus: Bus, result: RunResult) -> None:
    print("═" * 100)
    print("通信轨迹（每一步都是往某个 mailbox 投递一条消息）")
    print("═" * 100)
    print(f"{'#':>3} {'step':>4} {'rnd':>4}  {pad('from', 15)}{pad('→ to', 15)}"
          f"{pad('topic', 13)}payload")
    print("─" * 100)
    for r in bus.log:
        if "from" not in r:
            continue
        flag = ""
        if r.get("control"):
            flag = " [控制]"
        elif r.get("error"):
            flag = f" [拒收:{r['error']}]"
        print(f"{r['seq']:>3} {r['step']:>4} {r.get('round', 0):>4}  "
              f"{pad(r['from'], 15)}{pad('→ ' + r['to'], 15)}"
              f"{pad(r['topic'], 13)}{one_line(r['body'], 30)}{flag}")
    print("─" * 100)
    n_msg = sum(1 for r in bus.log if "from" in r)
    print(f"消息 {n_msg} 条 · 调度步数 {result.steps} · 干净收尾 {result.ok}")
    print(f"终止归因 {result.reason.value:<10} │ 余留未处理 {result.residual}"
          f" │ 解析失败丢弃 {result.dropped} │ 超轮次拒收 {result.rejected}")
    if bus.residual_detail():
        print(f"余留分布 {bus.residual_detail()}")
    for e in result.parse_errors[:8]:
        print(f"    ! {e}")
    print()


def print_isolation(bus: Bus) -> None:
    print("═" * 96)
    print("上下文隔离验证（每个 agent 只持有与自己相关的历史）")
    print("═" * 96)
    for name in bus.order:
        a = bus.agents[name]
        roles = [t["role"] for t in a.context]
        print(f"\n[{name}]  context {len(a.context)} 条 "
              f"(user {roles.count('user')} / assistant {roles.count('assistant')})")
        for t in a.context:
            tag = "recv" if t["role"] == "user" else "send"
            head = one_line(t["content"].split("\n", 1)[0], 62)
            print(f"    {tag:<5} {head}")
    print("\n" + "─" * 96)
    print("三者没有任何一条共享历史：producer 不知道 escalate，critic 不知道 escalate，")
    print("orchestrator 从未见过 producer 的推理过程。隔离是结构性事实，不是提示词约定。")
    print()


def print_verdict(bus: Bus) -> None:
    print("═" * 96)
    print("裁判定稿（编排者的最后一条回复，散文部分不参与路由）")
    print("═" * 96)
    print(strip_fence(bus.agents["orchestrator"].last_reply))
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Agent 间通信的最小可运行骨架")
    p.add_argument("--base-url", help="OpenAI 兼容端点，如 http://127.0.0.1:11434/v1")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--api-key", default="")
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="推理模型会把思维链也算进去，给小了会拿到空 content（默认 4096）")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--json", metavar="PATH", help="导出完整事件日志")
    args = p.parse_args()

    if args.base_url:
        print(f"[mode] 真实模型 {args.model} @ {args.base_url}")
        print(f"       max_tokens={args.max_tokens} temperature={args.temperature}\n")
        factory = lambda name: OpenAICompatLLM(  # noqa: E731
            args.base_url, args.model, args.api_key,
            temperature=args.temperature, max_tokens=args.max_tokens, label=name)
    else:
        print("[mode] 离线 mock（剧本化回复，用于验证协议本身，不消耗 token）\n")
        factory = scripted

    bus = build_demo(factory)
    try:
        # 入口：把目标投给编排者。kickoff 会绕过 quorum 屏障立刻跑第一步。
        bus.kickoff("orchestrator", Message(
            sender="@user", recipient="orchestrator", topic="goal", body=TASK))
        # max_round=3：让 Message.round 真正生效，超过第 3 轮的消息一律拒收。
        result = bus.run(max_steps=40, max_round=3)
    except LLMError as e:
        print(f"\n[模型报错] {e}\n")
        print("提示：本机 27B 级量化模型实测约 5 tok/s，一轮三方对抗要十几分钟，")
        print("      不适合用来跑拮抗。宿主 agent 自带的子代理用的是云端模型，")
        print("      速度快一个数量级且天然上下文隔离 —— 优先用那个。")
        print("      见技能 adversarial-subagent-review。")
        return 2

    print_trace(bus, result)
    print_isolation(bus)
    print_verdict(bus)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"run": {"reason": result.reason.value, "steps": result.steps,
                               "residual": result.residual, "dropped": result.dropped,
                               "rejected": result.rejected, "ok": result.ok},
                       "log": bus.log,
                       "contexts": {n: bus.agents[n].context for n in bus.order}},
                      f, ensure_ascii=False, indent=2)
        print(f"事件日志已写入 {args.json}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
