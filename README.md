# 共享隐藏状态下自适应查询泄露审计

对两份相邻数据（编号 0、1，共用 `n` 个零基隐藏状态）上的确定性自适应
查询策略做精确审计。全部计算使用 `fractions.Fraction`，无浮点误差。

## 用法

```bash
python3 -I audit.py <input.json> <output.json>
```

* 成功：把结果原子写入 `output.json`（替换旧文件），并在 stdout 回显；
  退出码 0。
* 失败：保留旧 `output.json` 不变，stdout 输出单个错误 JSON，退出码 1：
  ```json
  {"path": "$.K[1][0][2][3]", "message": "..."}
  ```
  语法错误路径固定为 `"$"`；下标均为零基。

处理不可信输入目录时使用 `python3 -I`（隔离模式），数据目录中植入的
`json.py` 等模块不会被加载。

## 输入

UTF-8 JSON，且仅含字段 `init`、`K`、`A`、`O`、`H`、`r`：

| 字段 | 形状 / 类型 | 约束 |
| --- | --- | --- |
| `A` | 非空字符串数组 | 互异的非空 ASCII 字母数字动作标识 |
| `O` | 非空字符串数组 | 互异的非空 ASCII 字母数字输出标识 |
| `init` | `[2, n]` | `p/q` 字符串；每份初始分布行和为 1 |
| `K` | `[2, |A|, n, |O|, n]` | `p/q` 字符串；`K[d][a][s]` 对 `(o,s')` 求和为 1 |
| `H` | 非负整数 | 布尔值不接受 |
| `r` | `p/q` 字符串 | 值 ≥ 1，无上界（区别于概率） |

概率字符串要求分子为非负整数、分母为正整数、值在 `[0,1]`；拒绝小数、
符号、空白与重复/未知字段。

## 输出

```json
{"direction": [0, 1], "delta": "1/2", "policy": [0, [null, null]]}
```

* `direction`：最优方向编号对；两个方向平局时取 `[0, 1]`。
* `delta`：目标值 `Σ_w max(P_d(w) − r·P_e(w), 0)` 的精确约分分数，
  分子分母均为十进制整数（零轮等情形为 `"0/1"`）。
* `policy`：策略节点 `[action, children]`，`action` 为零基动作下标，
  `children` 严格按 `O` 原序排列；轮数耗尽或双方概率均为零的分支为
  `null`，其余分支继续。动作平局取最小下标。

## 算法（`leakage_audit/planner.py`）

策略树的每个节点同时维护两份数据的前缀总质量 `m_d` 与归一化信念
`b_d`。观察 `o` 后的量由前向合并得到，自动合并产生同一输出串的全部
隐藏路径：

```
m'_d(o)  = m_d · Σ_s b_d[s] · Σ_s' K[d][a][s][o][s']
b'_d[s'] = (Σ_s b_d[s]·K[d][a][s][o][s']) / (m'_d(o)/m_d)
```

叶节点贡献 `max(m_d − r·m_e, 0)`；自底向上做 Bellman 递归，每个节点
枚举全部动作、对各输出子节点的值求和后取最大。状态用
`(剩余轮数, 质量二元组, 信念二元组)` 记忆化；Fraction 精确表示保证
相同信念状态的键严格相等。

## 验收（小树全枚举）

`leakage_audit/tests/` 内的验收测试枚举**全部**确定性策略树（内部节点
数 `(no^H−1)/(no−1)`，每节点任选动作），独立用前向合并逐串计算目标
值，再与 Bellman 规划器逐方向比对，并断言规划器返回的策略本身属于最
优策略集合。覆盖单/多状态、多动作、深度 1–3、隐藏路径合并、null 分支
规则、方向/动作平局、严格校验与 CLI 文件语义，共 42 个用例：

```bash
python3 -m unittest leakage_audit.tests.test_audit -v
# 隔离模式直接执行：
python3 -I leakage_audit/tests/test_audit.py
```

## 目录

```
audit.py                          入口（原子写、错误 JSON、退出码）
leakage_audit/
  parser.py                       严格 UTF-8 JSON 解析、重复字段拒绝
  validation.py                   字段/形状/类型/概率校验与模型构建
  planner.py                      前向合并 + Bellman 记忆化规划
  errors.py                       ValidationError / DuplicateKeyError
  tests/
    brute_force.py                全枚举与逐串评估（验收用）
    model_factory.py              测试模型构造
    test_audit.py                 42 个验收用例
```
