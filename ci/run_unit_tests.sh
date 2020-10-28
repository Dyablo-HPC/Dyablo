set -xe

cd build
# Display test stdout/err when failing
export CTEST_OUTPUT_ON_FAILURE=1
make dyablo-test
