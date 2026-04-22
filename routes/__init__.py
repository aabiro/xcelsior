"""Route modules for Xcelsior API."""

from routes.admin import router as admin_router
from routes.agent import router as agent_router
from routes.artifacts import router as artifacts_router
from routes.auth import router as auth_router
from routes.autoscale import router as autoscale_router
from routes.billing import router as billing_router
from routes.chat import router as chat_router
from routes.cloudburst import router as cloudburst_router
from routes.compliance import router as compliance_router
from routes.events import router as events_router
from routes.gpu import router as gpu_router
from routes.health import router as health_router
from routes.hosts import router as hosts_router
from routes.inference import router as inference_router
from routes.instances import router as instances_router
from routes.jurisdiction import router as jurisdiction_router
from routes.terminal import router as terminal_router
from routes.marketplace import router as marketplace_router
from routes.mfa import router as mfa_router
from routes.notifications import router as notifications_router
from routes.privacy import router as privacy_router
from routes.providers import router as providers_router
from routes.reputation import router as reputation_router
from routes.sla import router as sla_router
from routes.spot import router as spot_router
from routes.ssh import router as ssh_router
from routes.teams import router as teams_router
from routes.transparency import router as transparency_router
from routes.verification import router as verification_router
from routes.stripe_connect_v2 import router as stripe_connect_v2_router
from routes.static import router as static_router
from routes.volumes import router as volumes_router

ALL_ROUTERS = [
    admin_router,
    agent_router,
    artifacts_router,
    auth_router,
    autoscale_router,
    billing_router,
    chat_router,
    cloudburst_router,
    compliance_router,
    events_router,
    gpu_router,
    health_router,
    hosts_router,
    inference_router,
    instances_router,
    jurisdiction_router,
    terminal_router,
    marketplace_router,
    mfa_router,
    notifications_router,
    privacy_router,
    providers_router,
    reputation_router,
    sla_router,
    spot_router,
    ssh_router,
    static_router,
    stripe_connect_v2_router,
    teams_router,
    transparency_router,
    verification_router,
    volumes_router,
]
