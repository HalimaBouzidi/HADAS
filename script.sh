#==================================================================== NSGA-II (Evol_algo) ===============================================================================
python scripts/run.py  --problem hw_nas --algo nsga2 --n-init-sample 1 --batch-size 5 --n-iter 2 --n-seed 1 --n-process 1 --exp-name hw_nas
#==================================================================== DGEMO (MOBO) ===============================================================================
#python3 scripts/run.py --problem hw_nas --algo dgemo --n-init-sample 5 --batch-size 2 --n-iter 2 --n-seed 1 --n-process 1 --exp-name hw_nas
#==================================================================== USEMO-EI (MOBO) ===============================================================================
#python3 scripts/run.py  --problem hw_nas --algo usemo-ei --n-init-sample 5 --batch-size 2 --n-iter 2 --n-seed 1 --n-process 1 --exp-name hw_nas
