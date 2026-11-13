"""输入校验与模型构建。

校验覆盖：
* 字段集合必须恰为 init、K、A、O、H、r，拒绝未知/重复字段；
* 类型与形状严格校验（bool 不被接受为 int）；
* 概率必须是形如 ``"p/q"`` 的字符串，分子为非负整数、分母为正整数，
  且取值在 [0, 1]；
* 两个初始分布、每个 (数据, 动作, 当前状态) 的联合转移行和均为 1。

校验通过后返回内部模型（全部概率转为 :class:`fractions.Fraction`）。
"""

from fractions import Fraction

from .errors import ValidationError

_ALLOWED_KEYS = ("init", "K", "A", "O", "H", "r")


def _is_int(value):
    """JSON 中布尔值不是整数。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_list(value):
    return isinstance(value, list)


def _parse_fraction(value, path, name):
    """解析 p/q 字符串：分子为非负整数、分母为正整数（不限制上界）。"""
    if not isinstance(value, str):
        raise ValidationError(path, "%s 必须为 p/q 形式的字符串" % name)
    parts = value.split("/")
    if len(parts) != 2:
        raise ValidationError(path,
                              "%s 必须形如 p/q，且只含一个 '/'" % name)
    num_s, den_s = parts
    if not num_s or not den_s:
        raise ValidationError(path, "%s 的分子、分母均不能为空" % name)
    # 禁止符号、空白与非 ASCII 数字；isdigit 对 ASCII 数字串为真。
    if not (num_s.isdigit() and den_s.isdigit()):
        raise ValidationError(path,
                              "%s 的分子、分母必须为十进制整数字符串" % name)
    numerator = int(num_s)
    denominator = int(den_s)
    if denominator <= 0:
        raise ValidationError(path, "%s 分母必须为正整数" % name)
    return Fraction(numerator, denominator)


def _parse_probability(value, path):
    prob = _parse_fraction(value, path, "概率")
    if prob > 1:
        raise ValidationError(path, "概率取值必须在 [0, 1] 内")
    return prob


def _validate_identifiers(value, path, name):
    if not _is_list(value) or len(value) == 0:
        raise ValidationError(path, "%s 必须是非空数组" % name)
    seen = set()
    for i, item in enumerate(value):
        item_path = "%s[%d]" % (path, i)
        if not isinstance(item, str):
            raise ValidationError(item_path, "%s 元素必须为字符串" % name)
        if not item or not all("0" <= ch <= "9" or "a" <= ch <= "z"
                               or "A" <= ch <= "Z" for ch in item):
            raise ValidationError(item_path,
                                  "%s 元素必须为非空 ASCII 字母数字标识" % name)
        if item in seen:
            raise ValidationError(item_path, "%s 中存在重复标识 %r" % (name, item))
        seen.add(item)


def validate(obj):
    if not isinstance(obj, dict):
        raise ValidationError("$", "顶层结构必须是 JSON 对象")

    keys = set(obj.keys())
    allowed = set(_ALLOWED_KEYS)
    unknown = keys - allowed
    if unknown:
        name = sorted(unknown)[0]
        raise ValidationError("$.%s" % name, "未知字段 %r" % name)
    missing = allowed - keys
    if missing:
        name = sorted(missing)[0]
        raise ValidationError("$.%s" % name, "缺少必填字段 %r" % name)

    # A、O：非空、互异的 ASCII 字母数字标识数组
    _validate_identifiers(obj["A"], "$.A", "A")
    _validate_identifiers(obj["O"], "$.O", "O")
    n_actions = len(obj["A"])
    n_outputs = len(obj["O"])

    # H：非负整数（禁布尔）
    h = obj["H"]
    if not _is_int(h) or h < 0:
        raise ValidationError("$.H", "H 必须为非负整数")

    # r：至少为 1 的 p/q 字符串（无上界，区别于概率）
    r = _parse_fraction(obj["r"], "$.r", "r")
    if r < 1:
        raise ValidationError("$.r", "r 必须大于等于 1")

    # init：形状 [2, n]，n >= 1，行和为 1
    init_raw = obj["init"]
    if not _is_list(init_raw) or len(init_raw) != 2:
        raise ValidationError("$.init", "init 必须是形状 [2, n] 的数组")
    n = None
    init = []
    for d in range(2):
        row_path = "$.init[%d]" % d
        row = init_raw[d]
        if not _is_list(row) or len(row) == 0:
            raise ValidationError(row_path, "初始分布必须是长度 n >= 1 的数组")
        if n is None:
            n = len(row)
        elif len(row) != n:
            raise ValidationError(row_path, "两份数据的初始分布长度必须一致")
        probs = []
        total = Fraction(0)
        for s in range(n):
            p = _parse_probability(row[s], "%s[%d]" % (row_path, s))
            probs.append(p)
            total += p
        if total != 1:
            raise ValidationError(row_path, "初始分布概率和必须为 1，实际为 %s" % total)
        init.append(tuple(probs))

    # K：形状 [2, 动作数, n, 输出数, n]
    k_raw = obj["K"]
    k_path = "$.K"
    if not _is_list(k_raw) or len(k_raw) != 2:
        raise ValidationError(k_path, "K 必须是形状 [2, 动作数, n, 输出数, n] 的数组")

    kf = []
    for d in range(2):
        d_path = "%s[%d]" % (k_path, d)
        d_block = k_raw[d]
        if not _is_list(d_block) or len(d_block) != n_actions:
            raise ValidationError(d_path,
                                  "K 第一维下的动作数必须等于 A 的长度 %d" % n_actions)
        actions_block = []
        for a in range(n_actions):
            a_path = "%s[%d]" % (d_path, a)
            a_block = d_block[a]
            if not _is_list(a_block) or len(a_block) != n:
                raise ValidationError(a_path, "K 当前状态维长度必须等于 n=%d" % n)
            states_block = []
            for s in range(n):
                s_path = "%s[%d]" % (a_path, s)
                s_block = a_block[s]
                if not _is_list(s_block) or len(s_block) != n_outputs:
                    raise ValidationError(s_path,
                                          "K 输出维长度必须等于 O 的长度 %d" % n_outputs)
                outputs_block = []
                row_total = Fraction(0)
                for o in range(n_outputs):
                    o_path = "%s[%d]" % (s_path, o)
                    o_block = s_block[o]
                    if not _is_list(o_block) or len(o_block) != n:
                        raise ValidationError(o_path,
                                              "K 下一状态维长度必须等于 n=%d" % n)
                    next_probs = []
                    for sp in range(n):
                        p = _parse_probability(o_block[sp],
                                               "%s[%d]" % (o_path, sp))
                        next_probs.append(p)
                        row_total += p
                    outputs_block.append(tuple(next_probs))
                if row_total != 1:
                    raise ValidationError(
                        s_path,
                        "联合概率行和必须为 1，实际为 %s" % row_total)
                states_block.append(tuple(outputs_block))
            actions_block.append(tuple(states_block))
        kf.append(tuple(actions_block))

    return {
        "n": n,
        "n_actions": n_actions,
        "n_outputs": n_outputs,
        "init": tuple(init),
        "K": tuple(kf),
        "H": h,
        "r": r,
    }
