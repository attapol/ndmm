#!/bin/sh
julia -p 4 exp_scripts/train_tune_multideepnet.jl ndmm_config.json
