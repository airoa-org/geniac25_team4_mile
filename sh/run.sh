module load singularitypro
module load cuda

export SINGULARITY_TMPDIR=/home/group_25b505/group_4/members/koen/tmp/ # TODO: change to your own tmp directory
export target_dir="/home/group_25b505/group_4/members/koen/geniac25_team4_mile"

singularity shell --fakeroot --nv \
  --pwd /workspace/geniac25_team4_mile \
  --bind /home/group_25b505/group_4/datasets:/opt/processed \
  --bind "${target_dir}:/workspace/geniac25_team4_mile" \
  "${target_dir}/robot_mile.sif"
