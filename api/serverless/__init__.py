"""Serverless services and the independently packaged worker SDK."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo
    from serverless.service import ServerlessService, get_serverless_service

__all__ = ["ServerlessRepo", "ServerlessService", "get_serverless_service"]


def __getattr__(name: str):
    # The worker image contains only worker_sdk. Importing that subpackage must
    # not require the API's repository, database drivers or service dependencies.
    if name == "ServerlessRepo":
        from serverless.repo import ServerlessRepo

        return ServerlessRepo
    if name in ("ServerlessService", "get_serverless_service"):
        from serverless.service import ServerlessService, get_serverless_service

        return {
            "ServerlessService": ServerlessService,
            "get_serverless_service": get_serverless_service,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
