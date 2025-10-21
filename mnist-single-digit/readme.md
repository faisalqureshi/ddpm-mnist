Allocate an interactive session and setup enviornment

salloc --time=0:10:0 --mem-per-cpu=3G --ntasks=1 --account=def-fqureshi

OR

salloc --time=0:10:0 --mem=4000M --gpus-per-node=1 --ntasks=1 --account=def-fqureshi

and run

source setup.sh
