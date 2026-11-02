"""生成側のただ 1 つのエラーの型: 頼まれたとおりにできない呼び出しは SystemExit ではなく GenerationError を
投げる。呼んだ側がそれを実行のエラーとして記録し、続けられるようにするため。"""
from ..common import dbformat


class GenerationError(Exception):
    """生成の呼び出しを頼まれたとおりにできなかった。理由はメッセージにある。"""


class VersionError(GenerationError, dbformat.VersionError):
    """生成側が投げる dbformat.VersionError: `except GenerationError` で捕まえられ、dbformat.VersionError を
    捕まえるコードもそのまま動く。"""

    @classmethod
    def of(cls, ex):
        """同じエラーを生成側の VersionError にしたもの（メッセージと属性はそのまま）。"""
        out = cls.__new__(cls, *ex.args)
        out.args, out.what, out.found, out.reads = ex.args, ex.what, ex.found, ex.reads
        return out
