"""SSH secrets are encrypted at rest and revealed only on authorized detail reads."""
from security import decrypt_secret, encrypt_secret


def protect_ssh_status(status: dict) -> dict:
    stored = dict(status)
    password = stored.pop("root_password", "")
    if password:
        stored["root_password_encrypted"] = encrypt_secret(password)
    return stored


def present_ssh_status(status: dict, *, reveal: bool = False) -> dict:
    result = dict(status)
    encrypted = result.pop("root_password_encrypted", None)
    # Never return legacy plaintext through generic list/telemetry paths.
    legacy = result.pop("root_password", None)
    if reveal:
        result["root_password"] = decrypt_secret(encrypted) if encrypted else (legacy or "")
    return result
