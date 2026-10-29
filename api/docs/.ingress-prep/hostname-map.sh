#!/usr/bin/env bash
# Extract the mapping that silently moves: server_name -> (block, certificate),
# plus whichever block owns unmatched traffic.
#
# Diffing `nginx -T` output directly is the wrong comparison. It is thousands of
# lines, and adding a block shifts context everywhere — the presence of the new
# vhost drowns out the thing that matters, which is whether any *existing*
# hostname now resolves to a different block or a different certificate.
#
#   ./hostname-map.sh nginx-T.baseline.txt > before.txt
#   ssh prod 'sudo nginx -T' > after-raw.txt
#   ./hostname-map.sh after-raw.txt > after.txt
#   diff before.txt after.txt
#
# The only acceptable difference for the default-server change is one new line
# for the block that now owns unmatched traffic. Any *existing* hostname whose
# block or certificate changed means it has been re-homed — stop and roll back.
set -euo pipefail

SRC="${1:?usage: $0 <nginx -T output>}"

awk '
  /^[[:space:]]*server[[:space:]]*\{/ { depth++; if (depth==1) { names=""; cert=""; def="" } }
  depth>=1 && /^[[:space:]]*server_name[[:space:]]/ {
      line=$0; sub(/^[[:space:]]*server_name[[:space:]]+/,"",line); sub(/;.*/,"",line);
      names = (names=="" ? line : names " " line)
  }
  depth>=1 && /^[[:space:]]*ssl_certificate[[:space:]]/ {
      line=$0; sub(/^[[:space:]]*ssl_certificate[[:space:]]+/,"",line); sub(/;.*/,"",line);
      cert=line
  }
  depth>=1 && /^[[:space:]]*listen[[:space:]].*default_server/ { def=" [DEFAULT]" }
  /\}/ {
      if (depth==1 && names!="") {
          n=split(names, parts, /[[:space:]]+/)
          for (i=1;i<=n;i++) if (parts[i] != "") printf "%s -> %s%s\n", parts[i], (cert=="" ? "(no cert)" : cert), def
      }
      if (depth>0) depth--
  }
' "$SRC" | sort -u
