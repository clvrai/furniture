#! /bin/bash
# NOTE: this script assumes default name format of demos 
# which is <agent_name>_<furniture_name>_ABCD.pkl 
# ABCD being a 4 digit demo number

a_flag=false
n_flag=false
agent_name=''
n_demos=''
start_num=0
demo_dir='demos'
zip=true
help=''

print_usage() {
  	printf "usage: ./gen_all_demos.sh -a agent_name -n number_demos
  	Optional arguments:
	-s: start_num, ie the 'ABCD' in Sawyer_toy_table_ABCD
	-z: move and zip demos in batches of 100 after generation finishes
	-d: directory to save demos to, defaults to demos/\n"
}

while getopts 'a:n:s:d:hz' opt
do
  case "$opt" in
    a) agent_name=$OPTARG; a_flag=true ;;
    f) furniture_name=$OPTARG; f_flag=true ;;
    n) n_demos=$OPTARG; n_flag=true ;; 
    s) start_num=$OPTARG ;;
	z) zip=true;;
	d) demo_dir=$OPTARG ;;
    h) print_usage ;;
    :) pass ;;
    *) print_usage; exit 1;;
  esac
done
shift $((OPTIND -1))

if ! $a_flag; then
	echo -e "\e[31mPlease specify an agent_name\e[39m"
	print_usage
	exit 1
fi

if ! $n_flag; then
	echo -e "\e[31mPlease specify the number of demos\e[39m"
	print_usage
	exit 1
fi

declare -a furniture_names
readarray -t furniture_names < sawyer_demo_gen_candidates.txt
echo "$furniture_names"

let i=0
while (( ${#furniture_names[@]} > i )); do
    # printf "${furniture_names[i]}\n"
    echo "./scripts/gen_demos.sh -a ${agent_name} -f ${furniture_names[i]} -n ${n_demos}"
	eval "./scripts/gen_demos.sh -a ${agent_name} -f ${furniture_names[i]} -n ${n_demos}"
	#zip into bunches of 100
	wait
	for j in $( eval echo {0..$((n_demos/100 -1))}); do
		eval "cd demos"
		zip_range=$((start_num+j))
		eval "zip ${agent_name}_${furniture_names[i]}_0${zip_range}XX ${agent_name}_${furniture_names[i]}_0${zip_range}*"
		eval "cd .."
	done
	#move generated demos to new dir
	eval "mkdir -p ${demo_dir}/${agent_name}/${furniture_names[i]}/"
	eval "mv ${demo_dir}/${agent_name}_${furniture_names[i]}_****.pkl ${demo_dir}/${agent_name}/${furniture_names[i]}/"
	((i++))
done
