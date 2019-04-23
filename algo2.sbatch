#!/bin/bash
#
#SBATCH --partition=normal
#sbatch --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=1024
#SBATCH --output=stdout_algo2_%J.txt
#SBATCH --error=stderr_algo2_%J.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=algo2
#SBATCH --mail-user=dlschon@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/oucs044/project2/
#
#################################################
./algo2 1
#./algo2 4 
#./algo2 16 
#./algo2 25 
#./algo2 36 
#./algo2 64 
