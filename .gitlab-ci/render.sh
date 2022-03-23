set -xe

root=`pwd`

if [ $# -eq 0 ]
then
    echo "render.sh <pvsm filename local to build/dyablo/test/solver>"
    exit -1
fi
pvsm_filename=$1

cd build/dyablo/test/solver

if [ -x "$(command -v Xvfb)" ]
then
    echo "use Xvfb to emulate display"
    Xvfb :99 &
    # Set trap to kill Xvfb when script is over
    XVFB_PID=$!
    trap "kill ${XVFB_PID}" SIGINT SIGTERM EXIT
    export DISPLAY=:99
else
    echo "Xvfb not available using DISPLAY=${DISPLAY}"
fi
pvbatch --force-offscreen-rendering ${root}/.gitlab-ci/render_pvsm.py ${pvsm_filename} ${root}/render.png
