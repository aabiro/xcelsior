#!/usr/bin/env python3
"""Provision the free-tier bridge and host boundary; run as root before workers.

Does not start tenants, restart Docker, or change the default Docker bridge.
Re-run after a host reboot/firewall restore, before admitting free-tier work.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from free_tier_network import BRIDGE, CHAIN, GATEWAY, NETWORK, SUBNET, firewall_rules, privileged_iptables

def main():
    if os.geteuid():
        raise SystemExit("Run this provisioning command as root")
    probe = subprocess.run(["docker", "network", "inspect", NETWORK], capture_output=True, text=True)
    if probe.returncode:
        subprocess.run(["docker", "network", "create", "--internal", "--subnet", SUBNET, "--gateway", GATEWAY,
                        "-o", "com.docker.network.bridge.enable_icc=false", "-o", f"com.docker.network.bridge.name={BRIDGE}", NETWORK], check=True)
    else:
        network=json.loads(probe.stdout)[0]
        if not network.get('Internal') or network.get('Options',{}).get('com.docker.network.bridge.name')!=BRIDGE:
            raise SystemExit("Existing network has different settings; refusing to change it")
    iptables=privileged_iptables()
    if subprocess.run(iptables+["-S",CHAIN],capture_output=True).returncode:
        subprocess.run(iptables+["-N",CHAIN],check=True)
    # Populate before attaching the INPUT hook. Existing rules are not flushed.
    rules=firewall_rules()
    for rule in rules[1:]+rules[:1]:
        if subprocess.run(iptables+["-C"]+rule,capture_output=True).returncode:
            verb=['-I','INPUT','1']+rule[1:] if rule[0]=='INPUT' else ['-A']+rule
            subprocess.run(iptables+verb,check=True)
    print("Free-tier internal bridge and host firewall provisioned")

if __name__ == '__main__':
    main()
