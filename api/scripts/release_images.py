#!/usr/bin/env python3
"""Build the production images on a workstation and pin a release to GHCR digests."""

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_HOST = "ghcr.io"
REGISTRY_OWNER = "xcelsior-gpu"
REGISTRY = f"{REGISTRY_HOST}/{REGISTRY_OWNER}"
PLATFORM = "linux/amd64"
CONTEXTS = {"api": ".", "frontend": "api/frontend", "mcp": "api/mcp"}
SERVICES = {
    "api": ("api", "api-blue", "scheduler-worker", "bg-worker", "ssh-gateway", "ssh-gateway-blue"),
    "frontend": ("frontend",),
    "mcp": ("mcp", "mcp-blue"),
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def frontend_args(project: Path, env_file: Path | None = None) -> dict[str, str]:
    public_args = re.findall(
        r"^ARG (NEXT_PUBLIC_[A-Z0-9_]+|META_PIXEL_ID)(?:=([^\n]*)|$)",
        (project / "api/frontend/Dockerfile").read_text(),
        re.MULTILINE,
    )
    command = ["docker", "compose"]
    if env_file is not None:
        command.extend(("--env-file", str(env_file), "-f", "docker-compose.yml"))
    result = subprocess.run(
        [*command, "config", "--format", "json", "frontend"],
        cwd=project,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise ValueError("could not resolve the frontend's Compose build arguments")
    configured = json.loads(result.stdout)["services"]["frontend"]["build"]["args"]
    return {
        name: default if configured.get(name) is None else configured[name]
        for name, default in public_args
    }


def args_hash(args: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()


def build(push: bool, output: Path | None, env_file: Path) -> None:
    commit = git("rev-parse", "HEAD")
    if push:
        if output is None:
            raise ValueError("--push requires --manifest")
        if output.exists():
            raise ValueError("choose a new manifest path")
        # The deploy step will sync this same committed source tree.
        git("fetch", "origin", "main")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True
        )

    refs = {}
    with tempfile.TemporaryDirectory(prefix="xcelsior-release-") as directory:
        root = Path(directory)
        with (root / "source.tar").open("wb") as archive:
            subprocess.run(["git", "archive", commit], cwd=ROOT, stdout=archive, check=True)
        tree = root / "source"
        tree.mkdir()
        subprocess.run(["tar", "-xf", str(root / "source.tar"), "-C", str(tree)], check=True)
        public_args = frontend_args(tree, env_file)
        for image, relative in CONTEXTS.items():
            context = tree / relative
            metadata = root / f"{image}.json"
            tag = f"{REGISTRY}/{image}:{commit}"
            command = [
                "docker",
                "buildx",
                "build",
                "--platform",
                PLATFORM,
                "--tag",
                tag,
                "--metadata-file",
                str(metadata),
                "--label",
                f"org.opencontainers.image.revision={commit}",
                "--push" if push else "--load",
            ]
            env = os.environ.copy()
            if image == "frontend":
                # Buildx reads the explicitly selected values from its environment.
                env.update(public_args)
                for name in public_args:
                    command.extend(("--build-arg", name))
            subprocess.run([*command, str(context)], env=env, check=True)
            if push:
                digest = json.loads(metadata.read_text())["containerimage.digest"]
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                    raise ValueError(f"BuildKit returned an invalid digest for {image}")
                refs[image] = f"{REGISTRY}/{image}@{digest}"
    if push:
        with output.open("x") as artifact:
            json.dump(
                {"commit": commit, "images": refs, "frontend_args_sha256": args_hash(public_args)},
                artifact,
                indent=2,
            )
            artifact.write("\n")


def release_compose(path: Path) -> str:
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("commit") != git("rev-parse", "HEAD"):
        raise ValueError("the image manifest must match the current checkout's commit")
    images = manifest.get("images")
    if not isinstance(images, dict) or images.keys() != SERVICES.keys():
        raise ValueError("the image manifest must contain API, frontend, and MCP digests")
    fingerprint = manifest.get("frontend_args_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch("[0-9a-f]{64}", fingerprint):
        raise ValueError("the image manifest must contain a frontend build-argument fingerprint")
    lines = [f"# {fingerprint}", "services:"]
    for image, services in SERVICES.items():
        ref = images[image]
        expected = rf"{re.escape(REGISTRY)}/{image}@sha256:[0-9a-f]{{64}}"
        if not isinstance(ref, str) or not re.fullmatch(expected, ref):
            raise ValueError(f"expected a pinned {REGISTRY}/{image} digest")
        for service in services:
            lines.extend(
                (
                    f"  {service}:",
                    f"    image: {ref}",
                    "    build: !reset null",
                    "    pull_policy: never",
                )
            )
    return "\n".join(lines)


def stage_compose(state: Path, overlay: str) -> str:
    # Read the effective Compose options in place. The production env file and
    # the rendered service config may both contain credentials.
    first_line = overlay.partition("\n")[0]
    if first_line != f"# {args_hash(frontend_args(ROOT))}":
        raise ValueError("the frontend was built with different public settings than this host")
    result = subprocess.run(
        ["docker", "compose", "config", "--environment"], cwd=ROOT, text=True, capture_output=True
    )
    if result.returncode:
        raise ValueError("could not resolve the host's Compose file list")
    options = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    separator = options.get("COMPOSE_PATH_SEPARATOR") or os.pathsep
    configured = options.get("COMPOSE_FILE")
    if configured:
        files = configured.split(separator)
    else:
        files = ["docker-compose.yml"]
        for override in ("docker-compose.override.yml", "docker-compose.override.yaml"):
            if (ROOT / override).is_file():
                files.append(override)
                break
    candidate = state / f"release-{hashlib.sha256(overlay.encode()).hexdigest()}.yml"
    with tempfile.NamedTemporaryFile(mode="w", dir=state, delete=False) as temporary:
        temporary.write(overlay)
    path = Path(temporary.name)
    try:
        compiled = subprocess.run(
            ["docker", "compose", "--profile", "blue", "config", "--format", "json"],
            cwd=ROOT,
            env={**os.environ, "COMPOSE_FILE": separator.join([*files, str(path)])},
            text=True,
            capture_output=True,
        )
        if compiled.returncode:
            raise ValueError("the effective pull-only Compose configuration is invalid")
        services = json.loads(compiled.stdout)["services"]
        for image, names in SERVICES.items():
            expected_ref = rf"{re.escape(REGISTRY)}/{image}@sha256:[0-9a-f]{{64}}"
            for name in names:
                service = services[name]
                if (
                    "build" in service
                    or service.get("pull_policy") != "never"
                    or not re.fullmatch(expected_ref, service.get("image", ""))
                ):
                    raise ValueError(f"Compose did not resolve {name} to a pull-only image")
        path.replace(candidate)
        return separator.join([*files, str(candidate)])
    finally:
        path.unlink(missing_ok=True)


def staged_image_ref(image: str) -> str:
    references = subprocess.check_output(
        ["docker", "compose", "config", "--images", image],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).splitlines()
    pattern = rf"{re.escape(REGISTRY)}/{image}@sha256:[0-9a-f]{{64}}"
    pinned = [ref for ref in references if re.fullmatch(pattern, ref)]
    if len(pinned) != 1:
        raise ValueError(f"expected one pinned image for {image}")
    return pinned[0]


def verify_pulled_images(commit: str) -> None:
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("expected the full source commit")
    for image in SERVICES:
        revision = subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                '{{ index .Config.Labels "org.opencontainers.image.revision" }}',
                staged_image_ref(image),
            ],
            cwd=ROOT,
            text=True,
        ).strip()
        if revision != commit:
            raise ValueError(f"the pulled {image} image was built from a different commit")


def pull_token(image: str) -> str:
    login = subprocess.check_output(
        ["gh", "api", "--hostname", "github.com", "user", "--jq", ".login"], text=True
    ).strip()
    token = subprocess.check_output(["gh", "auth", "token", "--hostname", "github.com"]).strip()
    basic = base64.b64encode(login.encode() + b":" + token).decode()
    query = urlencode(
        {"service": REGISTRY_HOST, "scope": f"repository:{REGISTRY_OWNER}/{image}:pull"}
    )
    client = http.client.HTTPSConnection(REGISTRY_HOST, timeout=15)
    try:
        client.request("GET", f"/token?{query}", headers={"Authorization": f"Basic {basic}"})
        response = client.getresponse()
        if response.status != 200:
            raise ValueError(f"GHCR refused pull credentials for {image} (HTTP {response.status})")
        scoped = json.load(response).get("token")
        if not isinstance(scoped, str) or not scoped or any(char.isspace() for char in scoped):
            raise ValueError(f"GHCR returned an invalid pull token for {image}")
        return scoped
    finally:
        client.close()


class DockerSocket(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/var/run/docker.sock")


def pull_staged_image(image: str, token: str) -> None:
    if not token or any(char.isspace() for char in token):
        raise ValueError(f"missing or invalid pull token for {image}")
    reference = staged_image_ref(image)
    query = urlencode({"fromImage": reference, "platform": PLATFORM})
    auth = base64.urlsafe_b64encode(json.dumps({"registrytoken": token}).encode()).decode()
    client = DockerSocket("localhost", timeout=120)
    try:
        client.request("POST", f"/images/create?{query}", headers={"X-Registry-Auth": auth})
        response = client.getresponse()
        if response.status != 200:
            raise ValueError(f"Docker refused the {image} image pull (HTTP {response.status})")
        for line in response:
            event = json.loads(line)
            if "error" in event or "errorDetail" in event:
                # Docker/registry responses may reflect credential material.
                raise ValueError(f"Docker could not pull the pinned {image} image")
    finally:
        client.close()
    print(f"Pulled pinned {image} image")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    builder = actions.add_parser("build", help="build from HEAD on this machine")
    builder.add_argument("--push", action="store_true", help="publish to GHCR")
    builder.add_argument("--manifest", type=Path, help="write the resulting commit and digests")
    builder.add_argument(
        "--frontend-env",
        type=Path,
        default=Path(os.devnull),
        help="dotenv with the public production frontend build values",
    )
    renderer = actions.add_parser(
        "compose", help="validate a release manifest and render its override"
    )
    renderer.add_argument("manifest", type=Path)
    stager = actions.add_parser(
        "stage-compose", help="stage the validated override on the deployment host"
    )
    stager.add_argument("state", type=Path)
    verifier = actions.add_parser(
        "verify-pull", help="check the source labels on the staged images"
    )
    verifier.add_argument("commit")
    issuer = actions.add_parser("pull-token", help="send a short-lived GHCR token through a pipe")
    issuer.add_argument("image", choices=SERVICES)
    puller = actions.add_parser("pull", help="pull a staged image using its stdin registry token")
    puller.add_argument("image", choices=SERVICES)
    args = parser.parse_args()
    try:
        if args.action == "compose":
            print(release_compose(args.manifest))
        elif args.action == "stage-compose":
            print(stage_compose(args.state, sys.stdin.read()))
        elif args.action == "verify-pull":
            verify_pulled_images(args.commit)
        elif args.action == "pull-token":
            if sys.stdout.isatty():
                raise ValueError("pipe the pull token directly to the deployment host")
            print(pull_token(args.image))
        elif args.action == "pull":
            pull_staged_image(args.image, sys.stdin.read(16384).strip())
        else:
            if args.manifest is not None and not args.push:
                raise ValueError("--manifest requires --push")
            build(args.push, args.manifest, args.frontend_env.resolve())
    except (
        OSError,
        ValueError,
        KeyError,
        http.client.HTTPException,
        subprocess.CalledProcessError,
    ) as exc:
        raise SystemExit(f"release images: {exc}") from None


if __name__ == "__main__":
    main()
