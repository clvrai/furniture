#!/bin/bash
furniture_name=${1:-toy_table}
n_demos=${2:-200}
start_num=${3:-0}
n_cores=$(eval "nproc --all")
n_proc=$((n_cores/2))
demos_per_proc=$((n_demos/n_proc))
echo "n_cores=$n_cores"
echo "running $n_proc instances"

for i in $( eval echo {1..$n_proc})
do
 seed=$((i + 150))
 start_count=$((start_num+(demos_per_proc)*(i-1)))
eval "python -m env.furniture_sawyer_gen --render False --furniture_name $furniture_name\
 --record_vid False --furn_xyz_rand 0.02 --furn_rot_rand 3 --seed $seed --unity False \
 --start_count $start_count --n_demos=$demos_per_proc&"
done

wait 
rem=$((n_demos%$n_proc))
#collect remainder demos
start_count=$((start_num+(demos_per_proc)*(n_proc)))
if [ "$rem" -ne "0" ]; then
	eval "python -m env.furniture_sawyer_gen --render False --furniture_name $furniture_name\
	 --record_vid False --furn_xyz_rand 0.02 --furn_rot_rand 3 --seed $seed --unity False \
	 --start_count $start_count --n_demos=$rem&"
fi
