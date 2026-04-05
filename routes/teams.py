"""Routes: teams."""

import threading as _threading
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _get_current_user,
    broadcast_sse,
    log,
)
from scheduler import (
    ALERT_CONFIG,
    log,
)
from db import UserStore

router = APIRouter()


# ── Helper: _send_team_email ──

def _send_team_email(to_email: str, subject: str, body_text: str, cta_url: str | None = None, cta_label: str = "Go to Dashboard"):
    """Send a styled team notification email in a background thread. Best-effort.

    Matches the dark-theme style from frontend/src/emails/layout.tsx.
    """
    from scheduler import ALERT_CONFIG
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    cfg = ALERT_CONFIG
    if not cfg.get("email_enabled") or not cfg.get("smtp_host"):
        return

    # Build styled HTML matching the React email templates
    cta_html = ""
    if cta_url:
        cta_html = f"""
        <div style="text-align:center;margin:32px 0">
          <a href="{cta_url}" style="display:inline-block;background-color:#dc2626;color:#ffffff;padding:12px 32px;border-radius:8px;font-size:16px;font-weight:600;text-decoration:none">{cta_label}</a>
        </div>"""

    # Convert newlines in body to styled paragraphs
    paragraphs = "".join(
        f'<p style="color:#94a3b8;font-size:16px;line-height:1.6;margin:0 0 16px">{line}</p>'
        for line in body_text.strip().split("\n\n")
        if line.strip()
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/></head>
<body style="background-color:#0f172a;color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:0">
<div style="max-width:600px;margin:0 auto;padding:40px 24px">
  <div style="margin-bottom:32px">
    <span style="display:inline-block;width:36px;height:36px;line-height:36px;text-align:center;border-radius:8px;background-color:#dc2626;color:#fff;font-weight:800;font-size:20px;vertical-align:middle">X</span>
    <a href="https://xcelsior.ca" style="font-size:22px;font-weight:700;color:#f8fafc;text-decoration:none;vertical-align:middle;margin-left:12px">Xcelsior</a>
  </div>
  <h1 style="color:#f8fafc;font-size:24px;font-weight:700;line-height:1.3;margin:0 0 16px">{subject}</h1>
  {paragraphs}
  {cta_html}
  <hr style="border:none;border-top:1px solid #334155;margin:32px 0"/>
  <p style="color:#64748b;font-size:13px;line-height:1.5">
    Xcelsior Computing Inc. · Canada<br/>
    <a href="https://xcelsior.ca/privacy" style="color:#38bdf8;text-decoration:underline">Privacy Policy</a> ·
    <a href="https://xcelsior.ca/terms" style="color:#38bdf8;text-decoration:underline">Terms</a>
  </p>
</div>
</body></html>"""

    def _do_send():
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Xcelsior] {subject}"
            msg["From"] = cfg["email_from"]
            msg["To"] = to_email
            # Plain text fallback
            msg.attach(MIMEText(body_text, "plain"))
            # Styled HTML
            msg.attach(MIMEText(html, "html"))
            with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
                server.starttls()
                server.login(cfg["smtp_user"], cfg["smtp_pass"])
                server.send_message(msg)
            log.info("TEAM EMAIL SENT: %s -> %s", subject, to_email)
        except Exception as e:
            log.warning("TEAM EMAIL FAILED: %s -> %s | %s", subject, to_email, e)

    import threading
    threading.Thread(target=_do_send, daemon=True).start()


# ── Model: CreateTeamRequest ──

class CreateTeamRequest(BaseModel):
    name: str
    plan: str = "free"  # free | pro | enterprise


# ── Model: AddTeamMemberRequest ──

class AddTeamMemberRequest(BaseModel):
    email: str
    role: str = "member"  # admin | member | viewer


# ── Model: UpdateTeamMemberRoleRequest ──

class UpdateTeamMemberRoleRequest(BaseModel):
    role: str  # admin | member | viewer

@router.post("/api/teams", tags=["Teams"])
def api_create_team(body: CreateTeamRequest, request: Request):
    """Create a new team/organization. Creator becomes team admin."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team_id = f"team-{uuid.uuid4().hex[:8]}"
    max_members = {"free": 5, "pro": 25, "enterprise": 100}.get(body.plan, 5)

    team = {
        "team_id": team_id,
        "name": body.name,
        "owner_email": user["email"],
        "created_at": time.time(),
        "plan": body.plan,
        "max_members": max_members,
    }

    UserStore.create_team(team)
    UserStore.update_user(user["email"], {"team_id": team_id})
    broadcast_sse("team_created", {"team_id": team_id, "name": body.name})

    return {"ok": True, "team_id": team_id, "name": body.name, "plan": body.plan}

@router.get("/api/teams/me", tags=["Teams"])
def api_my_teams(request: Request):
    """Get teams the current user belongs to."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    teams = UserStore.get_user_teams(user["email"])
    return {"ok": True, "teams": teams}

@router.get("/api/teams/{team_id}", tags=["Teams"])
def api_get_team(team_id: str, request: Request):
    """Get team details including members."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    members = UserStore.list_team_members(team_id)
    # Verify the requester is a member
    if not any(m["email"] == user["email"] for m in members):
        raise HTTPException(403, "Not a member of this team")

    return {"ok": True, "team": team, "members": members}

@router.post("/api/teams/{team_id}/members", tags=["Teams"])
def api_add_team_member(team_id: str, body: AddTeamMemberRequest, request: Request):
    """Add a member to a team. Only team admins can add members."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    # Check requester is admin
    members = UserStore.list_team_members(team_id)
    requester = next((m for m in members if m["email"] == user["email"]), None)
    if not requester or requester["role"] != "admin":
        raise HTTPException(403, "Only team admins can add members")

    # Verify target user exists
    target = UserStore.get_user(body.email)
    if not target:
        raise HTTPException(404, f"User {body.email} not found")

    ok = UserStore.add_team_member(team_id, body.email, body.role)
    if not ok:
        raise HTTPException(400, "Team is at member capacity")

    broadcast_sse("team_member_added", {"team_id": team_id, "email": body.email})
    # Send email notification (best-effort, non-blocking)
    _send_team_email(
        body.email,
        f"You've been added to team {team['name']}",
        f"You've been added to the team \"{team['name']}\" on Xcelsior as a {body.role}.\n\nYou can now collaborate with your team, share billing, and manage GPU instances together.",
        cta_url="https://xcelsior.ca/dashboard/settings",
        cta_label="View Your Team",
    )
    return {"ok": True, "message": f"{body.email} added to team as {body.role}"}

@router.delete("/api/teams/{team_id}/members/{email}", tags=["Teams"])
def api_remove_team_member(team_id: str, email: str, request: Request):
    """Remove a member from a team. Admins can remove anyone; members can leave."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    members = UserStore.list_team_members(team_id)
    requester = next((m for m in members if m["email"] == user["email"]), None)
    if not requester:
        raise HTTPException(403, "Not a member of this team")

    # Non-admins can only remove themselves
    if requester["role"] != "admin" and email != user["email"]:
        raise HTTPException(403, "Only admins can remove other members")

    # Prevent removing the owner
    if email == team["owner_email"]:
        raise HTTPException(400, "Cannot remove team owner")

    UserStore.remove_team_member(team_id, email)
    # Send email notification (best-effort, non-blocking)
    _send_team_email(
        email,
        f"You've been removed from team {team['name']}",
        f"You've been removed from the team \"{team['name']}\" on Xcelsior.\n\nIf you believe this was a mistake, contact the team owner.",
        cta_url="https://xcelsior.ca/dashboard/settings",
        cta_label="Go to Dashboard",
    )
    return {"ok": True, "message": f"{email} removed from team"}

@router.patch("/api/teams/{team_id}/members/{email}", tags=["Teams"])
def api_update_team_member_role(team_id: str, email: str, body: UpdateTeamMemberRoleRequest, request: Request):
    """Update a team member's role. Only admins can change roles."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if body.role not in ("admin", "member", "viewer"):
        raise HTTPException(400, "Role must be admin, member, or viewer")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    members = UserStore.list_team_members(team_id)
    requester = next((m for m in members if m["email"] == user["email"]), None)
    if not requester or requester["role"] != "admin":
        raise HTTPException(403, "Only team admins can change roles")

    if email == team["owner_email"]:
        raise HTTPException(400, "Cannot change the team owner's role")

    if not UserStore.update_team_member_role(team_id, email, body.role):
        raise HTTPException(404, f"{email} is not a member of this team")

    return {"ok": True, "message": f"{email} role updated to {body.role}"}

@router.delete("/api/teams/{team_id}", tags=["Teams"])
def api_delete_team(team_id: str, request: Request):
    """Delete a team. Only the team owner can delete it."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    if team["owner_email"] != user["email"]:
        raise HTTPException(403, "Only the team owner can delete this team")

    UserStore.delete_team(team_id)
    broadcast_sse("team_deleted", {"team_id": team_id})
    return {"ok": True, "message": "Team deleted"}

