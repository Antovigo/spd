#!/bin/bash

cd ~/SPD/spd_alt

# export TMPDIR=/ephemeral/$USER
export TMPDIR=/mnt/nw/home/a.vigouroux

uv run spd-harvest /mnt/nw/home/a.vigouroux/SPD/batch_commands/css_targeted/harvest_config.yaml
