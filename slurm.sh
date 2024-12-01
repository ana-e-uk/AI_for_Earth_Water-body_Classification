#!/bin/bash -l        
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --mem=4g
#SBATCH --tmp=4g
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --mail-user=husse408@umn.edu 

# module load python3/3.8.3_anaconda2020.07_mamba 
# cd /users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification  # Youssef
cd /users/6/uribe055/AI_for_Earth_Water-body_Classification    # Ana
source ~/AI_EARTH/AI_for_Earth_Water-body_Classification/ai_venv_u/bin/activate
python src/train.py