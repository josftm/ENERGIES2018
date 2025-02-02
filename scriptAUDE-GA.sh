#$ -S /bin/bash
#$ -wd /home/agilson/jobs
#$ -q corta_multicore
#$ -pe smp 2-8
#$ -l virtual_free=24G
# $ -l slots=4
#
# Copio el fichero de entrada a un subdirectorio mio en /scratch
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi
mkdir -p /scratch/agilson/jobs/scr$1
cp ensembleScriptEvTree-RF-NN-GBM.R /scratch/agilson/jobs/scr$1
cp dataHW${2}PW24.csv /scratch/agilson/jobs/scr$1
cp modelEvtree_HW${2}PW24_p${1}.rda /scratch/agilson/jobs/scr$1
# Ahora que he preparado el entorno de trabajo
# en el nodo en el que se va a ejecutar mi programa, lo lanzo
#
export OMP_NUM_THREADS=$NSLOTS
module load R-3.4.2-Bioconductor
R --slave --no-save --max-ppsize 500000 --args $1 $2 < /scratch/agilson/jobs/scr$1/ensembleScriptEvTree-RF-NN-GBM.R
#
# Limpio el scratch
# Si el proceso hubiese dejado ficheros de salida que me interesan
# los copio antes a mi /home:
cp /scratch/agilson/jobs/scr$1/modelNN_HW${2}PW24_p${1}.rda /home/agilson/jobs
cp /scratch/agilson/jobs/scr$1/modelGBM_HW${2}PW24_p${1}.rda /home/agilson/jobs
cp /scratch/agilson/jobs/scr$1/modelRF_HW${2}PW24_p${1}.rda /home/agilson/jobs
rm -rf /scratch/agilson/jobs/scr$1