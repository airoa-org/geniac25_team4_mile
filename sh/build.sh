export SINGULARITY_TMPDIR=/home/group_25b505/group_4/members/koen/tmp/ # TODO: change to your own tmp directory
module load singularitypro
module load cuda

singularity build --fakeroot --nv --force robot_mile.sif robot_mile.def
