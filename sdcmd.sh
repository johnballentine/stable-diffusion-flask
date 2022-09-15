#!/bin/bash

# Runs txt2img StableDiffusion commands or generates the string to run

# Defaults
prompt='a photograph of an astronaut riding a horse'
ddim_steps=50
scale=7.5
seed=$[ $RANDOM % 1024 + 1 ]
run=false

while getopts p:n:s:x:r: flag
do
    case "${flag}" in
        p) prompt=${OPTARG};;
        n) ddim_steps=${OPTARG};;
        s) scale=${OPTARG};;
        x) seed=${OPTARG};;
        r) run=true;;
    esac
done

cmd="python scripts/txt2img.py --prompt \"$prompt\" --ddim_steps $ddim_steps --scale $scale --seed $seed --plms --n_samples 1 --skip_grid --n_iter 1 --outdir static/outputs"

if "$run"; then
    eval "$cmd"
else
    printf "\n\n"
    echo "python scripts/txt2img.py --prompt \"$prompt\" --ddim_steps $ddim_steps --scale $scale --seed $seed --plms --n_samples 1 --skip_grid --n_iter 1 --outdir static/outputs"
    printf "\n\n"
fi