#!/bin/bash
#
#SBATCH --partition=normal
#sbatch --nodes=1 
#SBATCH --ntasks=20
#SBATCH --time=00:30:00
#SBATCH --job-name=foxs_openmp
#SBATCH --output=%J_stdout.txt
#SBATCH --error=%J_stderr.txt
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/oucs044/project2
##################################################
./foxs 4 1024
