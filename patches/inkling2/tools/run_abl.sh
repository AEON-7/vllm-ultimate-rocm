#!/bin/bash
# usage: run_abl.sh mode...   (GPU1 only, worktree code)
cd ~/inkling/engines/aeon-cleanroom && source env.sh
export ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0 PYTHONPATH=$HOME/inkling/engines/aeon-cleanroom/wt-abl
unset AEON_ABLATE AEON_WEIGHT_OVERLAY AEON_VARIANT_DIR AEON_CAPTURE_DIR
cd /tmp
for m in "$@"; do
  case $m in inproc|inproc2) S=abl_$m.py;; *) S=abl_validate.py;; esac; timeout 5400 python ~/inkling/engines/aeon-cleanroom/aeon/$S $m > ~/inkling/engines/aeon-cleanroom/runs/abl/$m.log 2>&1
  echo "$m exit $?" >> ~/inkling/engines/aeon-cleanroom/runs/abl/status.txt
done
