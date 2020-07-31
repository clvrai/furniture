#! /bin/bash -x
v=$1

if [[ "$v" == "1" ]]; then
    furnitures=("table_dockstra_0279" "chair_agne_0007")
elif [[ "$v" == "2" ]]; then
    furnitures=("bench_bjursta_0210" "table_bjorkudden_0207")
elif [[ "$v" == "3" ]]; then
    furnitures=("table_lack_0825" "toy_table")
elif [[ "$v" == "4" ]]; then
    furnitures=("chair_ingolf_0650" "table_liden_0920")
fi

# furnitures=("table_dockstra_0279" "chair_agne_0007" "bench_bjursta_0210" "table_bjorkudden_0207" "table_lack_0825" "toy_table" "chair_ingolf_0650" "table_liden_0920")
# furnitures=("table_dockstra_0279" "chair_agne_0007")

for f in "${furnitures[@]}"; do
    echo "python -m run --algo bc --run_prefix ik_${f} --env furniture-sawyer-v0 --unity False --gpu 0 --record_video False --evaluate_interval 100 --max_episode_steps 500 --furniture_name ${f} --demo_path demos/old_Sawyer_${f}/Sawyer"
    python -m run --algo bc --run_prefix ik_${f} --env furniture-sawyer-v0 --unity False --gpu 0 --record_video False --evaluate_interval 100 --max_episode_steps 500 --furniture_name ${f} --demo_path demos/old_Sawyer_${f}/Sawyer
done

