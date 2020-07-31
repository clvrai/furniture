#! /bin/bash

furnitures=("table_dockstra_0279" "chair_agne_0007" "bench_bjursta_0210" "table_bjorkudden_0207" "table_lack_0825" "toy_table" "chair_ingolf_0650" "table_liden_0920")

for f in "${furnitures[@]}"; do
    python -m run --algo bc --run_prefix joint_${f} --env furniture-sawyer-v0 --unity False --gpu 0 --record_video False --evaluate_interval 100 --control_type impedance --demo_low_level True --max_episode_steps 2000 --furniture_name ${f} --demo_path demos/Sawyer_${f}/Sawyer
done

