"""确定性自适应策略的穷举枚举（仅用于验收测试）。

枚举全部深度恰为 H 的确定性策略树（每个内部节点可任选动作），用前向
合并逐串计算两份数据的完整输出串概率，直接累加目标函数。策略数量随
深度指数增长，只适合小树。
"""

from fractions import Fraction

from leakage_audit.planner import advance


def _cross_product(options_lists):
    """多个候选列表的笛卡尔积。"""
    if not options_lists:
        yield []
        return
    for head in options_lists[0]:
        for tail in _cross_product(options_lists[1:]):
            yield [head] + tail


def all_policies(n_actions, n_outputs, horizon):
    """生成全部确定性策略树（叶节点为 None）。"""
    if horizon == 0:
        return iter([None])

    def build(depth):
        if depth == 0:
            return [None]
        subtrees = build(depth - 1)
        level = []
        child_choices = [subtrees for _ in range(n_outputs)]
        for action in range(n_actions):
            for children in _cross_product(child_choices):
                level.append([action, children])
        return level

    return iter(build(horizon))


def evaluate(model, policy, direction):
    """按给定策略计算方向 (d, e) 的目标值（穷举所有输出串）。"""
    d, e = direction
    r = model["r"]
    init = model["init"]
    masses0 = (Fraction(1), Fraction(1))
    beliefs0 = (init[0], init[1])
    total = Fraction(0)

    def walk(node, masses, beliefs, depth):
        nonlocal total
        if node is None:
            # 叶节点：depth 轮已执行完，或两数据概率均为零。
            total += max(masses[d] - r * masses[e], Fraction(0))
            return
        action, children = node
        branches = advance(model, masses, beliefs, action)
        for o, child in enumerate(children):
            new_masses, new_beliefs = branches[o]
            walk(child, new_masses, new_beliefs, depth + 1)

    walk(policy, masses0, beliefs0, 0)
    return total


def brute_force_best(model, direction):
    """穷举全部策略，返回 (最优值, 达到最优值的策略集合)。"""
    best = None
    winners = []
    for policy in all_policies(model["n_actions"], model["n_outputs"],
                               model["H"]):
        value = evaluate(model, policy, direction)
        if best is None or value > best:
            best = value
            winners = [policy]
        elif value == best:
            winners.append(policy)
    return best, winners
