"""Routes: chat."""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from routes._deps import (
    _get_current_user,
    log,
)
from scheduler import (
    log,
)
from chat import (
    CHAT_API_KEY,
    append_message,
    build_system_prompt,
    check_chat_rate_limit,
    get_conversation_messages,
    get_or_create_conversation,
    get_user_conversations,
    record_feedback,
    stream_chat_response,
)
from privacy import redact_pii
from ai_assistant import FEATURE_AI_ASSISTANT, get_suggestions

router = APIRouter()


# ── Model: ChatRequest ──

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None

@router.post("/api/chat", tags=["Chat"])
async def api_chat(body: ChatRequest, request: Request):
    """Stream an AI chat response about Xcelsior via SSE."""
    client_ip = request.client.host if request.client else "unknown"

    if not check_chat_rate_limit(client_ip):
        raise HTTPException(429, "Chat rate limit exceeded. Please wait a moment.")

    if not CHAT_API_KEY:
        raise HTTPException(503, "Chat is not configured.")

    # Sanitise user input
    user_message = redact_pii(body.message)

    # Get or create conversation (persisted to SQLite)
    user = _get_current_user(request)
    user_email = user.get("email") if user else None
    conversation_id, history = get_or_create_conversation(
        body.conversation_id, ip=client_ip, user_email=user_email
    )

    # Build messages array
    system_prompt = build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # Track user message
    append_message(conversation_id, "user", user_message)

    async def _generate():
        full_response = []
        # Send conversation_id as first event
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id})}\n\n"
        try:
            async for token in stream_chat_response(messages):
                full_response.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            # Store assistant response
            append_message(conversation_id, "assistant", "".join(full_response))
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            log.error("Chat stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred. Please try again.'})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.get("/api/chat/suggestions", tags=["Chat"])
def api_chat_suggestions():
    """Return suggested starter questions for the chat widget."""
    return {
        "ok": True,
        "suggestions": [
            "How do I list my GPU on the marketplace?",
            "What are the trust tiers?",
            "How does billing work?",
            "How do I submit a job?",
        ],
    }

@router.get("/api/chat/history/{conversation_id}", tags=["Chat"])
def api_chat_history(conversation_id: str, request: Request):
    """Return message history for an existing conversation."""
    client_ip = request.client.host if request.client else "unknown"
    if not check_chat_rate_limit(client_ip):
        raise HTTPException(429, "Rate limit exceeded.")
    messages = get_conversation_messages(conversation_id)
    if messages is None:
        raise HTTPException(404, "Conversation not found or expired.")
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "messages": messages,
    }

@router.get("/api/chat/conversations", tags=["Chat"])
def api_chat_conversations(request: Request):
    """List recent conversations for the authenticated user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated.")
    email = user.get("email", "")
    if not email:
        raise HTTPException(401, "No email in session.")
    conversations = get_user_conversations(email)
    return {"ok": True, "conversations": conversations}


# ── Model: ChatFeedbackRequest ──

class ChatFeedbackRequest(BaseModel):
    message_id: str
    vote: str  # "up" or "down"

@router.post("/api/chat/feedback", tags=["Chat"])
def api_chat_feedback(body: ChatFeedbackRequest):
    """Record thumbs-up / thumbs-down feedback on a chat message."""
    if body.vote not in ("up", "down"):
        raise HTTPException(400, "Vote must be 'up' or 'down'.")
    record_feedback(body.message_id, body.vote)
    return {"ok": True}


# ── Helper: _require_ai_enabled ──

def _require_ai_enabled():
    if not FEATURE_AI_ASSISTANT:
        raise HTTPException(404, "AI assistant is not enabled.")


# ── Model: AiChatRequest ──

class AiChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    page_context: str = ""


# ── Model: AiConfirmRequest ──

class AiConfirmRequest(BaseModel):
    confirmation_id: str
    approved: bool

@router.post("/api/ai/chat", tags=["AI Assistant"])
async def api_ai_chat(body: AiChatRequest, request: Request):
    """Stream an AI assistant response with tool-calling support via SSE."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    user_id = user.get("user_id", user.get("email", ""))
    if not check_ai_rate_limit(user_id):
        raise HTTPException(429, "Rate limit exceeded. Please wait a moment.")

    # Get or create conversation
    conversation_id = body.conversation_id
    if conversation_id:
        conv = ai_get_conversation(conversation_id, user_id)
        if not conv:
            raise HTTPException(404, "Conversation not found")
    else:
        conversation_id = ai_create_conversation(user_id)

    return StreamingResponse(
        stream_ai_response(body.message, conversation_id, user, body.page_context),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.get("/api/ai/conversations", tags=["AI Assistant"])
def api_ai_list_conversations(request: Request, limit: int = 30):
    """List the current user's AI assistant conversations."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    user_id = user.get("user_id", user.get("email", ""))
    convs = ai_list_conversations(user_id, limit=limit)
    return {"ok": True, "conversations": convs}

@router.get("/api/ai/conversations/{conversation_id}", tags=["AI Assistant"])
def api_ai_get_conversation(conversation_id: str, request: Request):
    """Get messages for an AI assistant conversation."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    user_id = user.get("user_id", user.get("email", ""))
    messages = ai_get_messages(conversation_id, user_id)
    conv = ai_get_conversation(conversation_id, user_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return {"ok": True, "conversation": conv, "messages": messages}

@router.delete("/api/ai/conversations/{conversation_id}", tags=["AI Assistant"])
def api_ai_delete_conversation(conversation_id: str, request: Request):
    """Delete an AI assistant conversation."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    user_id = user.get("user_id", user.get("email", ""))
    deleted = ai_delete_conversation(conversation_id, user_id)
    if not deleted:
        raise HTTPException(404, "Conversation not found")
    return {"ok": True}

@router.post("/api/ai/confirm", tags=["AI Assistant"])
async def api_ai_confirm(body: AiConfirmRequest, request: Request):
    """Approve or reject a pending AI tool action."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    user_id = user.get("user_id", user.get("email", ""))
    if not check_ai_rate_limit(user_id):
        raise HTTPException(429, "Rate limit exceeded. Please wait a moment.")

    return StreamingResponse(
        execute_confirmed_action(body.confirmation_id, user, body.approved),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.get("/api/ai/suggestions", tags=["AI Assistant"])
def api_ai_suggestions(request: Request):
    """Get context-aware suggestion chips for the AI assistant."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return {"ok": True, "suggestions": get_suggestions(user)}

