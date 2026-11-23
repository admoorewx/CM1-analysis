#!/bin/bash
# Example SLURM job. Copy to jobs/local/ (gitignored) and fill in the
# placeholders there, so accounts, emails and paths stay out of the repo.
#SBATCH --job-name=cm1-inventory
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=cm1-inventory_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

set -euo pipefail
source activate cm1analysis

cm1plot inventory "$CM1_DATA/control/cm1out_000120.nc"
