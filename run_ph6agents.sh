#!/usr/bin/env bash
# Run PH6Agents local workflow controller.
# Usage: bash run_ph6agents.sh [task description]
#        bash run_ph6agents.sh --inspect
#        bash run_ph6agents.sh --memory
set -e
cd "$(dirname "$0")"
exec python3 ph6_agents.py "$@"
