"""Serverless inference endpoints — control-plane service package."""

from serverless.repo import ServerlessRepo
from serverless.service import ServerlessService, get_serverless_service

__all__ = ["ServerlessRepo", "ServerlessService", "get_serverless_service"]