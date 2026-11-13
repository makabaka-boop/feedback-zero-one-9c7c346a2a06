"""校验错误类型。

所有输入非法情形都抛出 :class:`ValidationError`，其 ``path`` 为形如
``"$"``、``"$.K"``、``"$.A[2]"``、``"$.K[1][0][2][3]"`` 的 JSON 路径，
下标均为零基。
"""


class ValidationError(Exception):
    def __init__(self, path, message):
        super().__init__(message)
        self.path = path
        self.message = message

    def as_payload(self):
        return {"path": self.path, "message": self.message}


class DuplicateKeyError(Exception):
    """对象中出现重复键（JSON 标准默认会静默覆盖，这里必须拒绝）。"""

    def __init__(self, key):
        super().__init__(key)
        self.key = key
