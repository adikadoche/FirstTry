#! /bin/bash
# Usage:    bash run_with_slurm.sh  JOB_NAME  SCRIPT_PATH  SCRIPT_ARG_1  SCRIPT_ARG_2  ...
# Example:  bash run_with_slurm.sh  lol_job  print_args.sh  AAA  BBB  CCC
# The script to run must begin with a she-bang (script header), e.g. #! /bin/bash
# Run `run_with_slurm.sh` from the desired working directory.

export JOB_NAME=$1
SCRIPT_PATH=$2
SCRIPT_PARAMS="${@:3:99999}"
export GIT_HASH="$(git rev-parse HEAD)"
LOG_DIR="slurm_logs"
SLURM_PARTITION="studentkillable"  # max 1 concurrent job per student


if [[ ${JOB_NAME} == *".sh"* ]]; then
  echo "woops! your first argument contains '.sh', did you forget to specify a slurm job name?"
  exit 1
fi

if [[ ${SCRIPT_PATH} != *".sh"* ]]; then
  echo "woops! your second argument doesn't contain '.sh', did you forget to specify a runnable script?"
  exit 1
fi

chmod u+rwx ${SCRIPT_PATH}

mkdir --parents ${LOG_DIR}

DATE=$(date +"%Y-%m-%d-%H-%M-%S-%N")
DATE=${DATE::23}
LOG_PATH=${LOG_DIR}/${DATE}_${JOB_NAME}_slurm_log.txt
TEMPFILE_PATH=$(tempfile)
echo ""

sbatch \
  --job-name=${JOB_NAME}  \
  --output=${LOG_PATH}  \
  --error=${LOG_PATH}  \
  --partition=${SLURM_PARTITION}  \
  --time=2880  \
  --signal=USR1@120  \
  --nodes=1  \
  --ntasks=1  \
  --gpus=1  \
  --account=gpu-students \
  --export JOB_NAME,GIT_HASH \
  --exclude="s-006"  \
  ${SCRIPT_PATH} ${SCRIPT_PARAMS}  |  tee ${TEMPFILE_PATH}

JOB_ID=$(grep -oP '\d+' ${TEMPFILE_PATH})
rm ${TEMPFILE_PATH}

if [[ ${JOB_ID} != "" ]]; then
  echo "running with slurm, log path ${LOG_PATH}
  to check job status, run: \"  scontrol show jobid -dd ${JOB_ID}  \"
  "
  sleep 1
  scontrol show jobid -dd ${JOB_ID}
  echo ""
fi
