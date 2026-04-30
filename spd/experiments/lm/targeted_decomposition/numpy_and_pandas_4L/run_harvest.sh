#!/bin/bash

cd ~/SPD/spd

export TMPDIR=/mnt/nw/home/a.vigouroux/workspace
export SPD_OUT_DIR=/mnt/nw/home/a.vigouroux/spd_out

uv run spd-harvest /mnt/nw/home/a.vigouroux/SPD/batch_commands/numpy/reference_4L/harvest_config.yaml
