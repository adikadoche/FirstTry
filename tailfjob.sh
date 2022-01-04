#! /bin/bash
# Show the stdout file of a running slurm job.
# Usage: catjob JOB_ID , or catjob JOB_NAME
# setup: run this line in bash:
#        mkdir -p ~/bin  &&  cp catjob.sh ~/bin  &&  chmod +x ~/bin/catjob.sh  &&  ln -s ~/bin/catjob.sh ~/bin/catjob  &&  echo 'PATH="${PATH}:~/bin"' >> ~/.bashrc  &&  source ~/.bashrc

set -e  # exit when any command fails - useful for invalid job ids

JOB_ID_OR_NAME=$1

if [[ "${JOB_ID_OR_NAME}" = "" ]]; then
  echo "Must specificy job name or job id"
  exit 1
fi

JOB_ROW=$(squeue | grep ${JOB_ID_OR_NAME} | tail -n 1)

if [[ "${JOB_ROW}" = "" ]]; then
  echo "Couldn't find job '${JOB_ID_OR_NAME}'"
  exit 1
fi

echo "=======================================================================================================
Showing output of job${JOB_ROW}
======================================================================================================="

JOB_ID=$(echo ${JOB_ROW} | awk '{print $1}')
JOB_INFO=$(scontrol show jobid ${JOB_ID})
LOG_PATH=$(echo ${JOB_INFO} | grep -Po "(?<=StdOut=).*? ")
tail -n 1000 -f ${LOG_PATH}