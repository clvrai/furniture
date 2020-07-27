#! /bin/bash
# NOTE: this script assumes default name format of demos 
# which is <agent_name>_<furniture_name>_ABCD.pkl 
# ABCD being a 4 digit demo number

a_flag=false
f_flag=false
n_flag=false
agent_name=''
furniture_name=''
n_demos=''
start_num=0
demo_dir='demos'
help=''

print_usage() {
  	printf "usage: ./gen_demos.sh -a agent_name -f furniture_name -n number_demos
  	Optional arguments:
	-s: start_num, ie the 'ABCD' in sawyer_toy_table_ABCD\n"
}

while getopts 'a:f:n:s:h' opt
do
  case "$opt" in
    a) agent_name=$OPTARG; a_flag=true ;;
    f) furniture_name=$OPTARG; f_flag=true ;;
    n) n_demos=$OPTARG; n_flag=true ;; 
    s) start_num=$OPTARG ;;
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

if ! $f_flag; then
	echo -e "\e[31mPlease specify a furniture_name\e[39m"
	print_usage
	exit 1
fi

if ! $n_flag; then
	echo -e "\e[31mPlease specify the number of demos\e[39m"
	print_usage
	exit 1
fi


n_cores=$(eval "nproc --all")
n_proc=$((n_cores/2))
demos_per_proc=$((n_demos/n_proc))
echo "n_cores=$n_cores"
echo "running $n_proc instances"


for i in $( eval echo {1..$n_proc})
do
 seed=$((i + 50))
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
	 --start_count $start_count --n_demos=$rem"
fi


