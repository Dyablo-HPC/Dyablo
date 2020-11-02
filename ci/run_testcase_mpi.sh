set -xe

root=`pwd`

if [ $# -lt 2 ]
then
    echo "run_testcase.sh <.ini filename local to build/dyablo/test/solver> <MPI proc count>"
    exit -1
fi
TESTCASE=$1
nmpi=$2

cd build/dyablo/test/solver
OMP_NUM_THREADS=4 mpirun -np $nmpi ./test_solver ${TESTCASE}
