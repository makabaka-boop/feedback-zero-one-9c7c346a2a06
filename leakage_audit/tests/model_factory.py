"""测试用模型构造辅助。"""

from fractions import Fraction


def F(text):
    return Fraction(text)


def make_model(init, k, actions, outputs, horizon, r):
    """以 Fraction 直接构造内部模型（跳过字符串校验）。"""
    init_f = tuple(tuple(F(x) for x in row) for row in init)

    kf = []
    for data_row in k:
        data_block = []
        for action_row in data_row:
            action_block = []
            for state_row in action_row:
                state_block = []
                for out_row in state_row:
                    state_block.append(tuple(F(x) for x in out_row))
                action_block.append(tuple(state_block))
            data_block.append(tuple(action_block))
        kf.append(tuple(data_block))

    return {
        "n": len(init[0]),
        "n_actions": len(actions),
        "n_outputs": len(outputs),
        "init": init_f,
        "K": tuple(kf),
        "H": horizon,
        "r": F(r),
    }


def single_state_model(dist0, dist1, actions=("a",), outputs=("x", "y"),
                       horizon=1, r="1"):
    """单状态 HMM：K[d][a][0][o][0] = 输出概率。"""
    n_actions = len(actions)
    k = []
    for dist in (dist0, dist1):
        data_block = []
        for _ in range(n_actions):
            state_row = [[[p] for p in dist]]
            data_block.append(state_row)
        k.append(data_block)
    init = [["1"]] * 2
    return make_model(init, k, actions, outputs, horizon, r)
