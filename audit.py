#!/usr/bin/env python3
"""命令行入口。

用法::

    python3 -I audit.py <input.json> <output.json>

成功时把结果原子写入 output.json（替换旧文件）；任何校验/求解失败都
保留旧 output.json 不变，并以退出码 1 打印单个错误 JSON：
``{"path": "...", "message": "..."}``，语法错误的路径为 ``"$"``。
"""

import json
import os
import os.path
import sys

# 使脚本在 `python3 -I`（隔离模式，用于处理不可信输入目录）下仍能
# 导入与本脚本同目录的 leakage_audit 包。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leakage_audit import (  # noqa: E402
    DuplicateKeyError,
    ValidationError,
    optimize,
    parse_input,
    validate,
)


def _emit_error(path, message):
    sys.stdout.write(json.dumps({"path": path, "message": message},
                                ensure_ascii=False))
    sys.stdout.write("\n")


def main(argv):
    if len(argv) != 3:
        _emit_error("$", "用法: audit.py <input.json> <output.json>")
        return 1
    input_path, output_path = argv[1], argv[2]

    try:
        with open(input_path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        _emit_error("$", "无法读取输入文件: %s" % exc)
        return 1

    try:
        try:
            obj = parse_input(raw)
        except DuplicateKeyError as exc:
            raise ValidationError("$.%s" % exc.key, "对象中存在重复字段")
        model = validate(obj)
        result = optimize(model)
    except ValidationError as exc:
        _emit_error(exc.path, exc.message)
        return 1
    # 成功才替换输出文件：先写临时文件再原子 rename，失败保留旧文件。
    directory = os.path.dirname(os.path.abspath(output_path)) or "."
    tmp_path = None
    try:
        import tempfile

        fd, tmp_path = tempfile.mkstemp(prefix=".audit-", suffix=".tmp",
                                        dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, output_path)
    except OSError as exc:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        _emit_error("$", "无法写入输出文件: %s" % exc)
        return 1

    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
