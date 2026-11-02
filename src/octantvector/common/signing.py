"""公開した DB の版の署名（OctantSpecs contracts/、契約5（OctantHarvest to OctantCompass）署名）。

署名するバイト列は manifest.json を JSON として読み、JCS（RFC 8785）で正規化したものなので、字下げや改行は
関係ない。署名は Ed25519 で、隣の manifest.sig に置く:

  {"algorithm": "Ed25519", "keyId": <32 バイトの公開鍵の SHA-256 の先頭 16 桁の 16 進>,
   "signature": <64 バイトの署名の base64>}

manifest.sig はマニフェストの files に載せない。受け入れる公開鍵は trusted_keys.py に同梱する。

  keygen(path)          新しい秘密鍵を `path` に書く（上書きはしない）-> (公開鍵の base64, keyId)
  sign(folder, key)     秘密鍵のファイル `key` で `folder` に manifest.sig を書く -> そのパス
  inspect(folder, trust) manifest.json に誰が署名したか -> {"status", "keyId", "detail"}。例外は投げない。status は
                        "signed"（同梱の鍵か trust["keys"]（{keyId: 公開鍵の base64}）の鍵による）、
                        "unsigned"、"malformed"、"unknownKey"、"mismatch" のどれか

ホストでは、署名は作り手を示すだけで止めはしない。apply_update は "signed" 以外なら警告し（SignatureWarning）、
その版を使う: DB はデータでコードを動かさず、ファイルはどのみち索引の SHA-256 と照らすため。作り手の側は
もっと厳しい: build_index（generation/index.py）は "signed" の版だけを索引に載せる（allow_unsigned は開発用）。
"""
import base64
import hashlib
import json
import os
from pathlib import Path

ALGORITHM = "Ed25519"
SIG = "manifest.sig"


class SignatureWarning(UserWarning):
    """manifest.sig がない、壊れている、知らない鍵による、または合わないのに使った DB の版。"""


def _canonical(folder):
    import rfc8785
    return rfc8785.dumps(json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8")))


def key_id(public_raw):
    return hashlib.sha256(public_raw).hexdigest()[:16]


def _public_raw(private):
    from cryptography.hazmat.primitives import serialization
    return private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def keygen(path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    private = Ed25519PrivateKey.generate()
    pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # 鍵は上書きしない
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    raw = _public_raw(private)
    return base64.b64encode(raw).decode(), key_id(raw)


def sign(folder, key_path):
    from cryptography.hazmat.primitives import serialization
    private = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
    sig = {"algorithm": ALGORITHM, "keyId": key_id(_public_raw(private)),
           "signature": base64.b64encode(private.sign(_canonical(folder))).decode()}
    p = Path(folder) / SIG
    p.write_text(json.dumps(sig, indent=1) + "\n")
    return p


def check_trust(trust):
    """apply_update の `trust` から (keys, allow_unsigned)（None: 同梱の鍵だけ）。"""
    from . import trusted_keys
    t = dict(trust or {})
    unknown = set(t) - {"keys", "allowUnsigned"}
    if unknown:
        raise ValueError(f"trust: unknown keys {', '.join(sorted(unknown))}")
    extra = dict(t.get("keys") or {})
    for kid, pub in extra.items():
        try:
            raw = base64.b64decode(pub, validate=True)
        except (ValueError, TypeError):
            raise ValueError(f"trust: key {kid} is not base64") from None
        if len(raw) != 32 or key_id(raw) != kid:
            raise ValueError(f"trust: key {kid} is not a 32-byte Ed25519 public key with that keyId")
    return {**trusted_keys.KEYS, **extra}, bool(t.get("allowUnsigned", False))


def inspect(folder, trust=None):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    keys, _ = check_trust(trust)
    p = Path(folder) / SIG

    def out(status, detail, kid=None):
        return {"status": status, "keyId": kid, "detail": detail}
    try:
        sig = json.loads(p.read_text())
        alg, kid, raw = sig["algorithm"], sig["keyId"], base64.b64decode(sig["signature"], validate=True)
    except FileNotFoundError:
        return out("unsigned", f"{folder}: no {SIG}")
    except (OSError, ValueError, KeyError, TypeError) as ex:
        return out("malformed", f"{p}: not a signature ({ex})")
    if alg != ALGORITHM:
        return out("malformed", f"{p}: algorithm {alg!r}, only {ALGORITHM} is accepted", kid)
    if kid not in keys:
        return out("unknownKey", f"{p}: signed by key {kid}, which this OctantVector does not know", kid)
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(keys[kid])).verify(raw, _canonical(folder))
    except InvalidSignature:
        return out("mismatch", f"{p}: the signature does not match manifest.json", kid)
    return out("signed", f"{p}: signed by key {kid}", kid)

