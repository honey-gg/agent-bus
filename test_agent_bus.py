#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_bus.py 的回归测试。纯标准库，两种跑法都行：

    python3 test_agent_bus.py
    python3 -m pytest test_agent_bus.py -q

这些用例不是凭空写的 —— 每一条都对应一次真实对抗中对抗方提出的、
经裁判原样复跑验证过的失败场景。**测试就是那次对抗的产物。**

对应关系：
    test_C1_*  消息卧在收件箱里被静默丢下
    test_C2_*  四种终止情形返回同一个可见信号，调用方无法归因
    test_C3_*  Message.round 定义了却从未被使用（死字段）
    test_C4_*  模型返回非法载荷时静默跳过，没有任何可观测失败
    test_C5_*  以上各条此前无任何测试覆盖
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_bus import (  # noqa: E402
    Agent,
    Bus,
    Message,
    StopReason,
    JSON_BLOCK,
    build_demo,
    scripted,
)

IDLE = "```json\n[]\n```"


def make(name="a", reply=IDLE, quorum=1):
    return Agent(name, "sys", lambda s, c: reply, quorum=quorum)


# ═══════════════════════════════════════════════════════════════════
# C1 无静默丢弃
# ═══════════════════════════════════════════════════════════════════


class TestC1NoSilentDrop(unittest.TestCase):
    def test_repro_quorum_never_satisfied_is_observable(self):
        """原始失败场景：quorum=2 但只来了 1 条消息，run() 静默返回，
        调用方看不到任何「有消息被丢下」的信号。"""
        bus = Bus()
        a = make(quorum=2)
        bus.register(a)
        bus.deliver(Message("@user", "a", "goal", "hi"), step=0)

        result = bus.run(max_steps=10)

        self.assertEqual(1, result.residual, "必须暴露余留消息数")
        self.assertEqual(StopReason.RESIDUAL, result.reason)
        self.assertFalse(result.ok, "有余留消息时不得报告为干净收尾")
        self.assertEqual({"a": 1}, bus.residual_detail())

    def test_unknown_recipient_is_counted(self):
        """真实故障形态：模型幻觉出一个不存在的收件人。
        投递失败必须计数，不能只写日志就算了。"""
        bus = Bus()
        bus.register(make("a", '```json\n{"to":"ghost","topic":"x","body":"y"}\n```'))
        bus.deliver(Message("@user", "a", "goal", "g"), step=0)

        result = bus.run(max_steps=10)
        self.assertEqual(1, result.rejected)
        self.assertFalse(result.ok)
        self.assertTrue(any(r.get("error") == "unknown_recipient" for r in bus.log))

    def test_pre_run_reject_is_not_lost(self):
        """run() 之外的投递（setup 阶段）也必须留下痕迹：
        日志里有记录，且不被 run() 的状态重置抹掉。"""
        bus = Bus()
        bus.register(make())
        bus.deliver(Message("a", "ghost", "x", "y"), step=0)
        self.assertEqual(1, bus.rejected)

        bus.deliver(Message("@user", "a", "goal", "g"), step=0)
        bus.run(max_steps=10)

        self.assertEqual(1, bus.rejected, "生命周期计数器不得被 run() 清零")
        self.assertTrue(any(r.get("error") == "unknown_recipient" for r in bus.log))

    def test_clean_convergence_has_zero_residual(self):
        bus = Bus()
        bus.register(make())
        bus.deliver(Message("@user", "a", "goal", "g"), step=0)
        result = bus.run(max_steps=10)
        self.assertEqual(0, result.residual)
        self.assertTrue(result.ok)


# ═══════════════════════════════════════════════════════════════════
# C2 终止可归因
# ═══════════════════════════════════════════════════════════════════


class TestC2StopReason(unittest.TestCase):
    def test_repro_four_reasons_are_distinguishable(self):
        """原始失败场景：自然收敛与余留未处理产生完全相同的可见信号
        (ret=1, halted=False)，调用方无法区分。"""

        # NATURAL
        b1 = Bus()
        b1.register(make())
        b1.deliver(Message("@user", "a", "goal", "g"), 0)

        # HALT
        b2 = Bus()
        b2.register(make(reply='```json\n{"to": "@halt", "topic": "done", "body": "x"}\n```'))
        b2.deliver(Message("@user", "a", "goal", "g"), 0)

        # MAX_STEPS：两个 agent 无限互发
        ping = '```json\n{"to": "pong", "topic": "t", "body": "x"}\n```'
        pong = '```json\n{"to": "ping", "topic": "t", "body": "x"}\n```'
        b3 = Bus()
        b3.register(make("ping", ping))
        b3.register(make("pong", pong))
        b3.deliver(Message("@user", "ping", "goal", "g"), 0)

        # RESIDUAL
        b4 = Bus()
        b4.register(make(quorum=2))
        b4.deliver(Message("@user", "a", "goal", "g"), 0)

        reasons = [
            b1.run(max_steps=10).reason,
            b2.run(max_steps=10).reason,
            b3.run(max_steps=5).reason,
            b4.run(max_steps=10).reason,
        ]
        self.assertEqual(
            [StopReason.NATURAL, StopReason.HALT, StopReason.MAX_STEPS, StopReason.RESIDUAL],
            reasons,
        )
        self.assertEqual(4, len(set(reasons)), "四种情形必须两两可区分")

    def test_max_steps_exposes_step_count(self):
        ping = '```json\n{"to": "pong", "topic": "t", "body": "x"}\n```'
        pong = '```json\n{"to": "ping", "topic": "t", "body": "x"}\n```'
        bus = Bus()
        bus.register(make("ping", ping))
        bus.register(make("pong", pong))
        bus.deliver(Message("@user", "ping", "goal", "g"), 0)
        result = bus.run(max_steps=5)
        self.assertEqual(StopReason.MAX_STEPS, result.reason)
        self.assertGreaterEqual(result.steps, 5, "必须报告真实跑了多少步，便于判断是否被截断")


# ═══════════════════════════════════════════════════════════════════
# C3 协议字段必须真被使用
# ═══════════════════════════════════════════════════════════════════


class TestC3RoundIsLive(unittest.TestCase):
    def test_repro_round_enters_diagnostics(self):
        """原始失败场景：round=42 与 round=0 产生完全相同的日志与轨迹。"""
        bus = Bus()
        bus.register(make())
        bus.deliver(Message("@user", "a", "goal", "g", round=42), step=0)
        self.assertIn("round", bus.log[-1], "round 必须进入诊断日志")
        self.assertEqual(42, bus.log[-1]["round"])

    def test_repro_round_limit_rejects_and_counts(self):
        """round 必须参与收敛判据：超限的消息被拒收并计数。"""
        bus = Bus(max_round=3)
        bus.register(make("a", '```json\n{"to":"b","topic":"t","body":"x","round":9}\n```'))
        bus.register(make("b"))
        bus.deliver(Message("@user", "a", "goal", "g"), 0)

        result = bus.run(max_steps=10)

        self.assertEqual(1, result.rejected, "超轮次消息必须被拒收并计数")
        self.assertEqual(0, len(bus.agents["b"].inbox))
        self.assertTrue(any(r.get("error") == "round_limit_exceeded" for r in bus.log))

    def test_round_within_limit_passes(self):
        bus = Bus(max_round=3)
        bus.register(make("a", '```json\n{"to":"b","topic":"t","body":"x","round":3}\n```'))
        bus.register(make("b"))
        bus.deliver(Message("@user", "a", "goal", "g"), 0)
        result = bus.run(max_steps=10)

        self.assertEqual(0, result.rejected, "等于上限属于允许，不得拒收")
        # 注意断言在投递日志上，不在收件箱上：b 会被正常唤醒并消费掉那条消息，
        # 所以 run() 结束时收件箱本来就该是空的。
        delivered = [r for r in bus.log if r.get("from") == "a" and r.get("to") == "b"]
        self.assertEqual(1, len(delivered))
        self.assertNotIn("error", delivered[0])


# ═══════════════════════════════════════════════════════════════════
# C4 非法载荷必须可观测
# ═══════════════════════════════════════════════════════════════════


class TestC4ParseFailureObservable(unittest.TestCase):
    def _run(self, reply_text):
        bus = Bus()
        bus.register(make("a", reply_text))
        bus.register(make("critic"))
        bus.deliver(Message("@user", "a", "goal", "g"), 0)
        return bus, bus.run(max_steps=10)

    def test_repro_three_malformed_shapes(self):
        """原始失败场景：三种非法载荷全部静默跳过，error 记录数 0。"""
        cases = {
            "JSON语法错误": '```json\n{"to": "critic", "topic": "x", body: 缺引号}\n```',
            "合法JSON但缺to": '```json\n{"topic": "x", "body": "y"}\n```',
            "纯散文无json块": "我觉得这个方案没问题，就这样吧。",
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                bus, result = self._run(text)
                self.assertEqual(0, len(bus.agents["critic"].inbox))
                self.assertGreater(result.dropped, 0, f"{label} 必须被计数")
                self.assertTrue(result.parse_errors, f"{label} 必须留下错误明细")
                self.assertFalse(result.ok, f"{label} 不得报告为干净收尾")

    def test_legit_empty_emit_is_not_a_drop(self):
        """关键区分：合法的「什么都不发」不得被误报为解析失败。"""
        bus, result = self._run(IDLE)
        self.assertEqual(0, result.dropped)
        self.assertEqual([], result.parse_errors)

    def test_step_event_records_dropped(self):
        bus, _ = self._run("纯散文")
        steps = [r for r in bus.log if r.get("event") == "step"]
        self.assertTrue(steps)
        self.assertGreater(steps[-1]["dropped"], 0)


# ═══════════════════════════════════════════════════════════════════
# C5 端到端回归 + 解析单元测试
# ═══════════════════════════════════════════════════════════════════


class TestC5EndToEnd(unittest.TestCase):
    def test_demo_run_is_clean(self):
        """完整的红蓝对抗剧本必须干净收尾：显式 halt、无余留、无丢弃。"""
        bus = build_demo(scripted)
        bus.kickoff("orchestrator", Message("@user", "orchestrator", "goal", "T"))
        result = bus.run(max_steps=40, max_round=3)

        self.assertEqual(StopReason.HALT, result.reason)
        self.assertEqual(0, result.residual)
        self.assertEqual(0, result.dropped)
        self.assertEqual(0, result.rejected)
        self.assertTrue(result.ok)
        self.assertEqual(8, sum(1 for r in bus.log if "from" in r))

    def test_contexts_are_isolated(self):
        """隔离是结构性事实：三者不得共享历史。"""
        bus = build_demo(scripted)
        bus.kickoff("orchestrator", Message("@user", "orchestrator", "goal", "T"))
        bus.run(max_steps=40)

        p = {t["content"] for t in bus.agents["producer"].context}
        c = {t["content"] for t in bus.agents["critic"].context}
        o = {t["content"] for t in bus.agents["orchestrator"].context}
        self.assertEqual(set(), p & c)
        self.assertEqual(set(), p & o)
        self.assertEqual(set(), c & o)


# ═══════════════════════════════════════════════════════════════════
# 第 2 轮对抗审查抓出的缺陷（修复本身引入的，初版测试没覆盖）
# ═══════════════════════════════════════════════════════════════════


class TestN1OkDoesNotTrustHalt(unittest.TestCase):
    """N1：初版 RunResult.ok 对 HALT 无条件返回 True，
    于是「@halt + 余留未处理」被误报为干净收尾，把 C1 的可观测性又冲掉了。"""

    def test_halt_with_residual_is_not_ok(self):
        bus = Bus()
        bus.register(make("a", quorum=2))                     # 永远凑不满，留 1 条
        bus.register(make("b", '```json\n{"to":"@halt","topic":"t","body":"x"}\n```'))
        bus.deliver(Message("@user", "a", "goal", "g"), 0)
        bus.deliver(Message("@user", "b", "goal", "g"), 0)

        result = bus.run(10)

        self.assertEqual(StopReason.HALT, result.reason)
        self.assertEqual(1, result.residual)
        self.assertFalse(result.ok, "HALT 且有余留时不得报告干净收尾")

    def test_halt_with_dropped_is_not_ok(self):
        bus = Bus()
        bus.register(make("a", "```json\n{bad}\n```"))
        bus.register(make("b", '```json\n{"to":"@halt","topic":"t","body":"x"}\n```'))
        bus.deliver(Message("@user", "a", "goal", "g"), 0)
        bus.deliver(Message("@user", "b", "goal", "g"), 0)

        result = bus.run(10)

        self.assertEqual(StopReason.HALT, result.reason)
        self.assertEqual(1, result.dropped)
        self.assertFalse(result.ok, "HALT 且有载荷被丢弃时不得报告干净收尾")

    def test_clean_halt_is_ok(self):
        bus = Bus()
        bus.register(make("a", '```json\n{"to":"@halt","topic":"t","body":"x"}\n```'))
        bus.deliver(Message("@user", "a", "goal", "g"), 0)
        self.assertTrue(bus.run(10).ok)

    def test_max_steps_and_residual_are_never_ok(self):
        ping = '```json\n{"to":"pong","topic":"t","body":"x"}\n```'
        pong = '```json\n{"to":"ping","topic":"t","body":"x"}\n```'
        b = Bus()
        b.register(make("ping", ping))
        b.register(make("pong", pong))
        b.deliver(Message("@user", "ping", "goal", "g"), 0)
        self.assertFalse(b.run(max_steps=5).ok)


class TestN2RunIsRepeatable(unittest.TestCase):
    """N2：run() 不重置每轮状态。后果有二：计数跨轮累加；
    上一轮的 halted=True 让 `while not self.halted` 直接短路，第二轮空转。"""

    def test_counts_do_not_accumulate_across_runs(self):
        bus = Bus()
        bus.register(make("a", "```json\n{bad1}\n```"))
        bus.deliver(Message("@user", "a", "goal", "g1"), 0)
        r1 = bus.run(max_steps=10)
        self.assertEqual(1, r1.dropped)

        bus.agents["a"].llm = lambda s, c: "```json\n{bad2}\n```"
        bus.deliver(Message("@user", "a", "goal", "g2"), 0)
        r2 = bus.run(max_steps=10)

        self.assertEqual(1, r2.dropped, "计数不得跨轮累加")
        self.assertEqual(1, r2.steps, "步数必须从 0 重新起算")

    def test_previous_halt_does_not_short_circuit_next_run(self):
        bus = Bus()
        bus.register(make("a", '```json\n{"to":"@halt","topic":"t","body":"x"}\n```'))
        bus.deliver(Message("@user", "a", "goal", "g1"), 0)
        self.assertEqual(StopReason.HALT, bus.run(10).reason)

        bus.agents["a"].llm = lambda s, c: IDLE
        bus.deliver(Message("@user", "a", "goal", "g2"), 0)
        r2 = bus.run(10)

        self.assertNotEqual(StopReason.HALT, r2.reason,
                            "上一轮的 halted=True 不得短路这一轮")
        self.assertEqual(1, r2.steps, "第二轮必须真正跑了，而不是空转返回")


class TestParseUnit(unittest.TestCase):
    def test_parse_accepts_array_and_single(self):
        a = make()
        msgs, errs = a._parse(
            'x\n```json\n[{"to":"p","topic":"t","body":"1"},{"to":"q","topic":"t","body":"2"}]\n```'
        )
        self.assertEqual(["p", "q"], [m.recipient for m in msgs])
        self.assertEqual([], errs)

    def test_parse_returns_errors_not_silence(self):
        a = make()
        for bad in ["没有任何代码块", "```json\n{坏}\n```", '```json\n{"topic":"x"}\n```']:
            with self.subTest(bad=bad[:20]):
                msgs, errs = a._parse(bad)
                self.assertEqual([], msgs)
                self.assertTrue(errs, "每一种失败形态都必须产出一条错误")

    def test_json_block_regex_handles_multiline(self):
        self.assertEqual(1, len(JSON_BLOCK.findall("```json\n{\n\"a\":1\n}\n```")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
