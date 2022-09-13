set -xe

cd build/dyablo/unit_test
# Display test stdout/err when failing
export CTEST_OUTPUT_ON_FAILURE=1
make test
