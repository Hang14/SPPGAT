#!/bin/bash
#SBATCH -J chn_senti_corp
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=end
#SBATCH --mail-user=none@shanghaitech.edu.cn
#SBATCH -N 1
#SBATCH -t 10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/public/home/huhang/SPPGAT/results/chn_senti_corp/%j.out
#SBATCH --error=/public/home/huhang/SPPGAT/results/chn_senti_corp/%j.err
cd /public/home/huhang/SPPGAT
echo "$SLURM_JOB_ID">>results/"$SLURM_JOB_NAME"/main.out
bash scripts/run_csc.sh