export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Circuit_Probing/
export CONFIG_DIR=configs/Grokking/DAS/DAS/b2
for file in ${CONFIG_DIR}/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/data/mlepori/projects/Circuit_Probing/src/${CONFIG_DIR}/${JOBNAME}.yaml


    sbatch -J $JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/src/run.script
done

