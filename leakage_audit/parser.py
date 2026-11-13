"""严格的 UTF-8 JSON 解析。

* 语法错误统一以路径 ``"$"`` 上报；
* 任何层级的对象出现重复字段都拒绝；
* 输入必须以 UTF-8 编码；
* 顶层必须是 JSON 对象。
"""

import json

from .errors import DuplicateKeyError, ValidationError


def _reject_duplicates(pairs):
    seen = set()
    for key, _value in pairs:
        if key in seen:
            raise DuplicateKeyError(key)
        seen.add(key)
    return dict(pairs)


def parse_input(raw_bytes):
    """解析输入字节串，返回顶层 dict。"""
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise ValidationError("$", "输入必须为 UTF-8 编码的字节流")
    try:
        text = bytes(raw_bytes).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("$", "输入不是合法的 UTF-8 编码: %s" % exc)

    try:
        obj = json.loads(text, object_pairs_hook=_reject_duplicates)
    except DuplicateKeyError:
        # 重复键路径在调用方（入口）映射到具体的 "$.<键>"。
        raise
    except json.JSONDecodeError as exc:
        raise ValidationError("$", "JSON 语法错误: %s" % exc.msg)

    if not isinstance(obj, dict):
        raise ValidationError("$", "顶层结构必须是 JSON 对象")
    return obj
