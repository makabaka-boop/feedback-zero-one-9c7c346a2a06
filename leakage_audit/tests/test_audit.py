"""验收测试。

覆盖：
1. 题目给定示例（单状态、一轮、r=1，delta=1/2，direction=[0,1]）；
2. 零轮；
3. 小树全枚举对照（含多状态隐藏路径合并、多动作、深度 2/3）；
4. 策略结构：children 按 O 原序、null 分支规则、动作平局最小下标；
5. 方向选择与 r>1 的非对称情形；
6. 严格校验：重复/未知字段、形状、类型、概率合法性；
7. CLI：成功替换输出、失败保留旧文件、语法错误路径为 $。
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

# 支持在 `python3 -I` 隔离模式下直接运行本测试文件。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from leakage_audit import (
    DuplicateKeyError,
    ValidationError,
    optimize,
    parse_input,
    plan_direction,
    validate,
)
from leakage_audit.tests.brute_force import (
    all_policies,
    brute_force_best,
    evaluate,
)
from leakage_audit.tests.model_factory import make_model, single_state_model

AUDIT = os.path.join(ROOT, "audit.py")


def parse_json(text):
    return json.loads(text)


class ExampleTests(unittest.TestCase):
    def test_given_example(self):
        # 单状态单动作、一轮：[3/4,1/4] 对 [1/4,3/4]，r=1
        model = single_state_model(
            ["3/4", "1/4"], ["1/4", "3/4"], horizon=1, r="1")
        result = optimize(model)
        self.assertEqual(result["direction"], [0, 1])
        self.assertEqual(result["delta"], "1/2")
        self.assertEqual(result["policy"], [0, [None, None]])

    def test_zero_rounds(self):
        model = single_state_model(
            ["3/4", "1/4"], ["1/4", "3/4"], horizon=0, r="1")
        result = optimize(model)
        # 零轮只有空串，两数据概率均为 1：max(1-1, 0) = 0
        self.assertEqual(result["delta"], "0/1")
        self.assertEqual(result["direction"], [0, 1])
        self.assertIsNone(result["policy"])

    def test_r_greater_than_one_asymmetric(self):
        model = single_state_model(
            ["3/4", "1/4"], ["1/2", "1/2"], horizon=1, r="2")
        result = optimize(model)
        # dir (0,1): max(3/4-2*1/2,0)+max(1/4-2*1/2,0) = 0
        # dir (1,0): max(1/2-2*3/4,0)+max(1/2-2*1/4,0) = 0
        self.assertEqual(result["delta"], "0/1")

        model2 = single_state_model(
            ["1/2", "1/2"], ["3/4", "1/4"], horizon=1, r="2")
        result2 = optimize(model2)
        # dir (0,1): max(1/2-3/2,0)+max(1/2-1/2,0) = 0
        # dir (1,0): max(3/4-1,0)+max(1/4-1/2,0) = 0
        self.assertEqual(result2["delta"], "0/1")

    def test_r_picks_direction(self):
        # d0=[2/3,1/3], d1=[1/3,2/3], r=2
        # dir(0,1): max(2/3-2/3,0)+max(1/3-4/3,0) = 0
        # dir(1,0): max(1/3-4/3,0)+max(2/3-2/3,0) = 0  -> 仍平局
        # 改用 r=3/2:
        # dir(0,1): max(2/3-1/2,0)+0 = 1/6
        # dir(1,0): 0+max(2/3-1/2,0) = 1/6 -> 还是平局（交换对称）
        # 非对称分布才能打破：
        model = single_state_model(
            ["3/4", "1/4"], ["1/4", "3/4"], horizon=1, r="3/2")
        # dir(0,1): max(3/4-3/8,0)+max(1/4-9/8,0)=3/8
        # dir(1,0): max(1/4-9/8,0)+max(3/4-3/8,0)=3/8 对称仍相等
        result = optimize(model)
        self.assertEqual(result["delta"], "3/8")
        self.assertEqual(result["direction"], [0, 1])

        model2 = single_state_model(
            ["4/5", "1/5"], ["1/2", "1/2"], horizon=1, r="2")
        # dir(0,1): max(4/5-1,0)+max(1/5-1,0) = 0
        # dir(1,0): max(1/2-8/5,0)+max(1/2-2/5,0) = 1/10
        result2 = optimize(model2)
        self.assertEqual(result2["direction"], [1, 0])
        self.assertEqual(result2["delta"], "1/10")


class BruteForceTests(unittest.TestCase):
    def _assert_matches_brute_force(self, model):
        for direction in ((0, 1), (1, 0)):
            bf_value, winners = brute_force_best(model, direction)
            dp_value, dp_policy = plan_direction(model, direction)
            self.assertEqual(dp_value, bf_value)
            # 规划器给出的策略必须本身是最优策略之一
            self.assertEqual(evaluate(model, dp_policy, direction), bf_value)
            self.assertIn(_canonical(dp_policy),
                          {_canonical(p) for p in winners})

    def test_single_state_two_outputs_depth_2(self):
        model = single_state_model(
            ["2/3", "1/3"], ["1/2", "1/2"], horizon=2, r="1")
        self._assert_matches_brute_force(model)

    def test_single_state_two_outputs_depth_3_r_2(self):
        model = single_state_model(
            ["3/4", "1/4"], ["1/2", "1/2"], horizon=3, r="2")
        self._assert_matches_brute_force(model)

    def test_two_states_two_actions_two_outputs(self):
        # n=2, na=2, no=2，含状态转移，隐藏路径必须合并。
        init = [["1/2", "1/2"], ["1/3", "2/3"]]
        # K[data][action][state][output][next_state]
        k = [
            # data 0
            [
                # action 0
                [
                    [["1/2", "0"], ["0", "1/2"]],    # s0: o0->s0, o1->s1
                    [["1/4", "1/4"], ["1/4", "1/4"]],  # s1
                ],
                # action 1
                [
                    [["1/3", "1/6"], ["1/6", "1/3"]],
                    [["1/2", "0"], ["0", "1/2"]],
                ],
            ],
            # data 1
            [
                [
                    [["1/4", "1/4"], ["1/4", "1/4"]],
                    [["0", "1/2"], ["1/2", "0"]],
                ],
                [
                    [["1/2", "0"], ["0", "1/2"]],
                    [["1/6", "1/3"], ["1/3", "1/6"]],
                ],
            ],
        ]
        model = make_model(init, k, ("a0", "a1"), ("x", "y"),
                           horizon=2, r="1")
        self._assert_matches_brute_force(model)

        model_r2 = make_model(init, k, ("a0", "a1"), ("x", "y"),
                              horizon=2, r="3/2")
        self._assert_matches_brute_force(model_r2)

    def test_enumeration_count(self):
        # 深度 H、na 动作、no 输出的满树策略数 = na^((no^H-1)/(no-1))
        policies = list(all_policies(2, 2, 2))
        self.assertEqual(len(policies), 2 ** 3)  # 3 个内部节点
        policies3 = list(all_policies(2, 2, 3))
        self.assertEqual(len(policies3), 2 ** 7)  # 7 个内部节点


def _canonical(node):
    if node is None:
        return None
    action, children = node
    return (action, tuple(_canonical(c) for c in children))


class MergingTests(unittest.TestCase):
    def test_hidden_paths_are_merged(self):
        # 两个隐状态都能产生同一输出，验证前向合并的串概率。
        # 无论当前状态：o0 必转移到 s0，o1 必转移到 s1（行和均为 1）。
        init = [["1/2", "1/2"], ["1/1", "0/1"]]
        # K[d][a][s][o][s']：o0 必到 s0、o1 必到 s1，每行和为 1
        transition = [
            [["1/1", "0/1"], ["0/1", "0/1"]],  # state 0: o0->s0
            [["0/1", "0/1"], ["0/1", "1/1"]],  # state 1: o1->s1
        ]
        k = [
            [transition],  # data 0, 单动作
            [transition],  # data 1
        ]
        model = make_model(init, k, ("a",), ("0", "1"), horizon=2, r="1")
        _v, policy = plan_direction(model, (0, 1))
        # 直接校验合并质量
        from leakage_audit.planner import advance
        masses = (Fraction(1), Fraction(1))
        beliefs = (model["init"][0], model["init"][1])
        branches = advance(model, masses, beliefs, 0)
        # 数据 0：初始 s0、s1 各 1/2；s0 必出 o0，s1 必出 o1
        # => o0、o1 的串概率各合并为 1/2
        self.assertEqual(branches[0][0][0], Fraction(1, 2))
        self.assertEqual(branches[1][0][0], Fraction(1, 2))
        # 前缀 o0 后数据 0 的信念必落在 s0
        self.assertEqual(branches[0][1][0], (Fraction(1), Fraction(0)))
        # 第二轮在 s0：o0 概率 1、o1 概率 0
        branches2 = advance(model, branches[0][0], branches[0][1], 0)
        self.assertEqual(branches2[0][0][0], Fraction(1, 2))
        self.assertEqual(branches2[1][0][0], Fraction(0))
        self.assertIsNotNone(policy)

    def test_hidden_paths_merge_partial(self):
        # 真正的多路径合并：s0 产生 o0 的总概率 3/4，s1 为 1/2，
        # 初始各 1/2 => o0 串概率 = 1/2*3/4 + 1/2*1/2 = 5/8。
        init = [["1/2", "1/2"], ["1/1", "0/1"]]
        k = [
            [[
                [["1/2", "1/4"], ["1/8", "1/8"]],   # s0: o0=3/4, o1=1/4
                [["1/4", "1/4"], ["1/4", "1/4"]],   # s1: o0=1/2, o1=1/2
            ]],
            [[
                [["1/2", "1/4"], ["1/8", "1/8"]],
                [["1/4", "1/4"], ["1/4", "1/4"]],
            ]],
        ]
        model = make_model(init, k, ("a",), ("0", "1"), horizon=1, r="1")
        from leakage_audit.planner import advance
        branches = advance(model, (Fraction(1), Fraction(1)),
                           (model["init"][0], model["init"][1]), 0)
        # 数据 0：1/2*3/4 + 1/2*1/2 = 5/8
        self.assertEqual(branches[0][0][0], Fraction(5, 8))
        # 数据 1 初始全在 s0：o0 概率 3/4
        self.assertEqual(branches[0][0][1], Fraction(3, 4))


class PolicyStructureTests(unittest.TestCase):
    def test_children_follow_O_order_and_null_rules(self):
        # 构造输出 1 在数据 0 下恒为零概率的模型（数据 1 非零 => 不剪枝）
        k = [
            [[[["1/1", "0/1"], ["0/1", "0/1"]]]],  # data0: 只有 o0
            [[[["1/2", "0/1"], ["1/2", "0/1"]]]],  # data1: o0/o1 各半
        ]
        init = [["1/1"], ["1/1"]]
        model = make_model(init, k, ("a",), ("o0", "o1"), horizon=2, r="1")
        _value, policy = plan_direction(model, (0, 1))
        action, children = policy
        self.assertEqual(action, 0)
        self.assertEqual(len(children), 2)
        # o0 分支：双方质量均非零且还剩 1 轮 => 必须继续（非 null）
        self.assertIsNotNone(children[0])
        # o1 分支：数据 0 质量为 0，但数据 1 质量 1/2 非零 => 继续
        self.assertIsNotNone(children[1])

    def test_both_zero_branch_is_null(self):
        # 某输出在两份数据下都不可能 => 该分支恒 null
        k = [
            [[[["1/1", "0/1"], ["0/1", "0/1"]]]],
            [[[["1/1", "0/1"], ["0/1", "0/1"]]]],
        ]
        init = [["1/1"], ["1/1"]]
        model = make_model(init, k, ("a",), ("o0", "o1"), horizon=2, r="1")
        _value, policy = plan_direction(model, (0, 1))
        _action, children = policy
        self.assertIsNotNone(children[0])
        self.assertIsNone(children[1])  # 双方概率均零

    def test_action_tie_picks_smallest_index(self):
        # 单状态、两个动作在两份数据下分布完全相同 => 动作平局取 0
        dist0 = ["1/2", "1/2"]
        dist1 = ["1/3", "2/3"]
        model = single_state_model(dist0, dist1,
                                   actions=("a0", "a1"), horizon=1, r="1")
        _value, policy = plan_direction(model, (0, 1))
        self.assertEqual(policy[0], 0)
        _value2, policy2 = plan_direction(model, (1, 0))
        self.assertEqual(policy2[0], 0)

    def test_full_depth_when_mass_remains(self):
        model = single_state_model(
            ["3/4", "1/4"], ["1/4", "3/4"], horizon=3, r="1")
        _value, policy = plan_direction(model, (0, 1))

        def depth(node):
            if node is None:
                return 0
            return 1 + max(depth(c) for c in node[1])

        self.assertEqual(depth(policy), 3)


class ValidationTests(unittest.TestCase):
    def _base_input(self):
        # n=2 隐状态、1 动作、2 输出；K[d][a][s][o][s'] 行和为 1
        return {
            "init": [["3/4", "1/4"], ["1/4", "3/4"]],
            "K": [
                [[
                    [["3/4", "0/1"], ["1/4", "0/1"]],
                    [["0/1", "1/4"], ["0/1", "3/4"]],
                ]],
                [[
                    [["1/4", "0/1"], ["3/4", "0/1"]],
                    [["0/1", "3/4"], ["0/1", "1/4"]],
                ]],
            ],
            "A": ["a"],
            "O": ["x", "y"],
            "H": 1,
            "r": "1/1",
        }

    def test_valid_input(self):
        model = validate(self._base_input())
        self.assertEqual(model["n"], 2)
        self.assertEqual(model["H"], 1)
        self.assertEqual(model["r"], Fraction(1))

    def test_unknown_field(self):
        obj = self._base_input()
        obj["extra"] = 1
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.extra")

    def test_missing_field(self):
        obj = self._base_input()
        del obj["r"]
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.r")

    def test_duplicate_key(self):
        text = ('{"init": [["1/1"],["1/1"]],'
                '"K": [[[[[["1/1"]]]]],[[[[["1/1"]]]]]],'
                ' "A": ["a"], "O": ["o"],'
                ' "H": 0, "r": "1/1", "H": 1}')
        with self.assertRaises(DuplicateKeyError):
            parse_input(text.encode("utf-8"))

    def test_bool_rejected_as_integer(self):
        obj = self._base_input()
        obj["H"] = True
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.H")

    def test_negative_horizon(self):
        obj = self._base_input()
        obj["H"] = -1
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.H")

    def test_bad_identifier(self):
        obj = self._base_input()
        obj["A"] = ["a-b"]
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.A[0]")

    def test_duplicate_identifier(self):
        obj = self._base_input()
        obj["O"] = ["x", "x"]
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.O[1]")

    def test_empty_arrays_rejected(self):
        obj = self._base_input()
        obj["A"] = []
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.A")

    def test_r_below_one(self):
        obj = self._base_input()
        obj["r"] = "1/2"
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.r")

    def test_r_wrong_format(self):
        obj = self._base_input()
        obj["r"] = "1"
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.r")

    def test_r_above_one_allowed(self):
        obj = self._base_input()
        obj["r"] = "5/2"
        model = validate(obj)
        self.assertEqual(model["r"], Fraction(5, 2))

    def test_probability_out_of_range(self):
        obj = self._base_input()
        obj["init"][0][0] = "5/4"
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.init[0][0]")

    def test_zero_denominator(self):
        obj = self._base_input()
        obj["init"][0][0] = "1/0"
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.init[0][0]")

    def test_non_numeric_probability(self):
        obj = self._base_input()
        obj["init"][0][0] = "x/1"
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.init[0][0]")

    def test_probability_not_string(self):
        obj = self._base_input()
        obj["init"][0][0] = 0.5
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.init[0][0]")

    def test_init_row_sum(self):
        obj = self._base_input()
        obj["init"] = [["1/2", "1/4"], ["1/4", "3/4"]]
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.init[0]")

    def test_k_row_sum(self):
        obj = self._base_input()
        # 破坏 s0 行：原行和 1，改后 o0+o1 = 1/2 + 1/4 = 3/4
        obj["K"][0][0][0][0] = ["1/2", "0/1"]
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.K[0][0][0]")

    def test_bad_k_shape(self):
        obj = self._base_input()
        obj["K"] = [obj["K"][0]]  # 缺一份数据
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.K")

    def test_bad_k_inner_shape(self):
        obj = self._base_input()
        # 下一状态维少一项（长度 1，应为 n=2）
        obj["K"][0][0][0][0] = ["1/1"]
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.K[0][0][0][0]")

    def test_init_shape_mismatch(self):
        obj = self._base_input()
        obj["init"] = [["1/1"], ["1/2", "1/2"]]
        with self.assertRaises(ValidationError) as ctx:
            validate(obj)
        self.assertEqual(ctx.exception.path, "$.init[1]")

    def test_syntax_error_path_is_root(self):
        with self.assertRaises(ValidationError) as ctx:
            parse_input(b"{not json")
        self.assertEqual(ctx.exception.path, "$")

    def test_top_level_must_be_object(self):
        with self.assertRaises(ValidationError) as ctx:
            parse_input(b"[1, 2]")
        self.assertEqual(ctx.exception.path, "$")

    def test_non_utf8(self):
        with self.assertRaises(ValidationError) as ctx:
            parse_input(b"\xff\xfe{}")
        self.assertEqual(ctx.exception.path, "$")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="audit-test-")

    def _write(self, name, content):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as fh:
            fh.write(content if isinstance(content, bytes)
                     else content.encode("utf-8"))
        return path

    def _run(self, input_path, output_path):
        proc = subprocess.run(
            [sys.executable, "-I", AUDIT, input_path, output_path],
            capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    def test_success_replaces_output(self):
        payload = {
            "init": [["1/1"], ["1/1"]],
            "K": [
                [[[["3/4"], ["1/4"]]]],
                [[[["1/4"], ["3/4"]]]],
            ],
            "A": ["a"],
            "O": ["x", "y"],
            "H": 1,
            "r": "1/1",
        }
        inp = self._write("in.json", json.dumps(payload))
        out = os.path.join(self.tmp, "out.json")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("OLD CONTENT")
        code, stdout, _err = self._run(inp, out)
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as fh:
            result = json.loads(fh.read())
        self.assertEqual(result["delta"], "1/2")
        self.assertEqual(result["direction"], [0, 1])
        self.assertEqual(json.loads(stdout)["delta"], "1/2")

    def test_failure_keeps_old_file(self):
        bad = {
            "init": [["1/1"], ["1/1"]],
            "K": [
                [[[["3/4"], ["1/4"]]]],
                [[[["1/4"], ["3/4"]]]],
            ],
            "A": ["a"],
            "O": ["x", "y"],
            "H": -1,
            "r": "1/1",
        }
        inp = self._write("bad.json", json.dumps(bad))
        out = os.path.join(self.tmp, "out.json")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("KEEP ME")
        code, stdout, _err = self._run(inp, out)
        self.assertEqual(code, 1)
        err = json.loads(stdout)
        self.assertEqual(err["path"], "$.H")
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "KEEP ME")

    def test_syntax_error_reports_root(self):
        inp = self._write("syntax.json", b"{broken")
        out = os.path.join(self.tmp, "out2.json")
        code, stdout, _err = self._run(inp, out)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)["path"], "$")
        self.assertFalse(os.path.exists(out))

    def test_duplicate_key_error_path(self):
        text = ('{"init": [["1/1"],["1/1"]],'
                '"K": [[[[[["1/1"]]]]],[[[[["1/1"]]]]]],'
                '"A": ["a"], "O": ["o"], "H": 0, "r": "1/1",'
                ' "A": ["b"]}')
        inp = self._write("dup.json", text)
        out = os.path.join(self.tmp, "out3.json")
        code, stdout, _err = self._run(inp, out)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)["path"], "$.A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
