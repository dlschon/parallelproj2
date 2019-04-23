#!/bin/bash
#
#SBATCH --partition=normal
#sbatch --nodes=1 
#SBATCH --ntasks=20
#SBATCH --mem=1024
#SBATCH --time=00:30:00
#SBATCH --job-name=foxs_openmp
#SBATCH --output=%J_stdout.txt
#SBATCH --error=%J_stderr.txt
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/oucs016/SIPE/proj2/
##################################################
./foxs 4 1024