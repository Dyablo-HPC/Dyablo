set -xe

root=`pwd`

if [ $# -eq 0 ]
then
    echo "run_testcase.sh <.ini filename local to build/dyablo/test/solver>"
    exit -1
fi
TESTCASE=$1

cd build/dyablo/test/solver
./test_solver ${TESTCASE}