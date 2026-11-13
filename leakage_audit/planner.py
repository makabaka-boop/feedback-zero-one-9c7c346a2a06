"""自适应策略优化（精确有理数）。

模型定义
--------
两份数据编号 d = 0, 1，共用 n 个零基隐藏状态。对给定数据 d、动作 a、
当前状态 s，``K[d][a][s][o][s']`` 是观察到输出 o 且下一状态为 s' 的
联合概率；对 (o, s') 求和为 1。

策略是一棵深度恰为 H 的决策树：内部节点保存所选动作，children 按 O 的
原序排列。规划器在每个节点上同时维护两份数据的

* 总质量 m_d = P_d(到当前节点的输出前缀)；
* 归一化信念 b_d[s] = P_d(当前隐藏状态 = s | 输出前缀)。

观察 o 后的量完全由前向合并得到（自动合并产生同一输出串的全部隐藏
路径）：

    m'_d(o)   = m_d * sum_s b_d[s] * sum_s' K[d][a][s][o][s']
    b'_d[s']  = (sum_s b_d[s] * K[d][a][s][o][s']) / (m'_d(o) / m_d)

方向 (d, e) 下，叶节点串 w 的贡献为 max(m_d - r * m_e, 0)。Bellman
递归即在每个节点枚举动作、对各输出子节点递归取值后求和，取最大者。
"""

from fractions import Fraction


def advance(model, masses, beliefs, action):
    """执行一个动作，返回每个输出分支的 (新质量二元组, 新信念二元组)。

    masses   : (m_0, m_1)，两份数据到达当前节点的前缀概率；
    beliefs  : (b_0, b_1)，各为长度 n 的 Fraction 元组；
    返回列表，下标与 O 的原序一致。
    """
    k = model["K"]
    n = model["n"]
    n_outputs = model["n_outputs"]
    branches = []
    for o in range(n_outputs):
        new_masses = []
        new_beliefs = []
        for d in range(2):
            weighted = [Fraction(0) for _ in range(n)]
            obs_prob = Fraction(0)
            belief = beliefs[d]
            for s in range(n):
                bs = belief[s]
                if bs == 0:
                    continue
                for sp in range(n):
                    joint = k[d][action][s][o][sp]
                    if joint:
                        weighted[sp] += bs * joint
                        obs_prob += bs * joint
            new_mass = masses[d] * obs_prob
            if obs_prob == 0:
                # 该数据在此分支概率为零：信念无定义，用零向量占位，
                # 后续不会再贡献任何质量。
                new_belief = tuple(Fraction(0) for _ in range(n))
            else:
                new_belief = tuple(x / obs_prob for x in weighted)
            new_masses.append(new_mass)
            new_beliefs.append(new_belief)
        branches.append((tuple(new_masses), tuple(new_beliefs)))
    return branches


def plan_direction(model, direction):
    """求指定方向上的最优值与策略树。

    返回 (value: Fraction, policy: 节点/None)。
    节点格式：``[action, [child_o0, child_o1, ...]]``，
    叶节点（轮数耗尽，或两数据质量均为零）为 ``None``。
    """
    d, e = direction
    horizon = model["H"]
    r = model["r"]
    init = model["init"]
    n = model["n"]

    root_beliefs = (init[0], init[1])
    root_masses = (Fraction(1), Fraction(1))
    memo = {}

    def solve(remaining, masses, beliefs):
        # 轮数耗尽：对完整输出串计分。
        if remaining == 0:
            value = max(masses[d] - r * masses[e], Fraction(0))
            return value, None

        # 双方概率均为零的分支：剪枝为 null，任何后续动作都不再产生贡献。
        if masses[0] == 0 and masses[1] == 0:
            return Fraction(0), None

        key = (remaining, masses, beliefs)
        cached = memo.get(key)
        if cached is not None:
            return cached

        best_value = None
        best_action = None
        best_children = None
        for action in range(model["n_actions"]):
            total = Fraction(0)
            children = [None] * model["n_outputs"]
            branches = advance(model, masses, beliefs, action)
            for o, (new_masses, new_beliefs) in enumerate(branches):
                child_value, child_node = solve(
                    remaining - 1, new_masses, new_beliefs)
                total += child_value
                children[o] = child_node
            # 动作下标递增枚举，严格大于才替换 => 平局自动取最小下标。
            if best_value is None or total > best_value:
                best_value = total
                best_action = action
                best_children = children

        result = (best_value, [best_action, best_children])
        memo[key] = result
        return result

    return solve(horizon, root_masses, root_beliefs)


def optimize(model):
    """两个方向各优化一次，按规则选方向并返回结果字典。"""
    candidates = []
    for direction in ((0, 1), (1, 0)):
        value, policy = plan_direction(model, direction)
        candidates.append((direction, value, policy))

    first = candidates[0]
    second = candidates[1]
    # 方向平局取 [0, 1]：仅当方向 1 严格更优时才选它。
    if second[1] > first[1]:
        chosen = second
    else:
        chosen = first

    direction, value, policy = chosen
    return {
        "direction": list(direction),
        "delta": "%d/%d" % (value.numerator, value.denominator),
        "policy": policy,
    }
