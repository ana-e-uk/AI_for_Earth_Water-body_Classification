#!/bin/bash -l   
#SBATCH -o /users/6/uribe055/AI_for_Earth_Water-body_Classification/out/out.output
#SBATCH -e /users/6/uribe055/AI_for_Earth_Water-body_Classification/out/out.error    
#SBATCH --time=24:00:00
#SBATCH -p msigpu
#SBATCH --ntasks=64
#SBATCH --mem=16g
#SBATCH --tmp=16g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=husse408@umn.edu 


# module load python3/3.8.3_anaconda2020.07_mamba 

# Youssef
# cd /users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification
# source ~/AI_EARTH/AI_for_Earth_Water-body_Classification/ai_venv_u/bin/activate

# Ana
cd /users/6/uribe055/AI_for_Earth_Water-body_Classification
source /users/6/uribe055/AI_for_Earth_Water-body_Classification/ai_venv_u/bin/activate

# Aylar
# cd /users/5/aghaa001/AI_for_Earth_Water-body_Classification
# source /users/5/aghaa001/AI_for_Earth_Water-body_Classification/ai_venv_ay/bin/activate

python src/train.py