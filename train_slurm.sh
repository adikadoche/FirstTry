#!/bin/bash
ADDITIONAL_PARAMS="${@:1:99999}"


PYTHONUNBUFFERED=1 /home/gamir/adiz/miniconda3/envs/torchGPU/bin/python \
  main.py \
  ${ADDITIONAL_PARAMS}