#!/bin/bash

cd ~/SPD/spd

# export TMPDIR=/ephemeral/$USER
export TMPDIR=/mnt/nw/home/a.vigouroux

uv run spd-harvest /mnt/nw/home/a.vigouroux/SPD/batch_commands/css/harvest_config_border-radius.yaml
