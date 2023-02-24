set +e
set -x
set -o pipefail

err_count=0
run_count=0
out_list=()
success_list=()

run_and_render(){
    ini_file=$1
    pvsm_file=$2
    out=$3
    stdout_filename=run_$out.txt
    png_filename=$out.png
    echo "Testing : Execution logs in artifact $stdout_filename"
    .gitlab-ci/run_testcase_mpi.sh $ini_file 4 &> $stdout_filename
    if [ $? -eq 0 ]
    then
	    echo "run ${ini_file} success"
        echo "Export image in artifact : $png_filename"
        .gitlab-ci/render.sh $pvsm_file
        mv render.png $png_filename
        success_list+=(1)
    else
	    echo "run ${ini_file} fail"
	    err_count=$((err_count+1))
        touch $png_filename
        success_list+=(0)
    fi 
    out_list+=($out)
    ini_list+=($ini_file)
    run_count=$((run_count+1))
}


run_and_render test_blast_3D_block.ini visu_blast_3D_block.pvsm blast_3d_block 
run_and_render test_blast_3D.ini visu_blast_3D_cell.pvsm blast_3d_cell

run_and_render test_blast_2D_block.ini visu_blast_2D_block.pvsm blast_2d_block 
run_and_render test_blast_2D.ini visu_blast_2D_cell.pvsm blast_2d_cell 

run_and_render test_riemann_2D.ini visu_riemann_2D_cell.pvsm riemann_2d_cell
run_and_render test_riemann_2D_block.ini visu_riemann_2D_block.pvsm riemann_2d_block

run_and_render test_gravity_spheres_3D.ini visu_gravity_spheres_3D.pvsm gravity_3D 

echo "${err_count}/${run_count} runs failed"


echo "<testsuites tests=\"${run_count}\" failures=\"${err_count}\" disabled=\"0\" errors=\"${err_count}\" name=\"AllTests\">" > report.xml
echo "  <testsuite name=\"DyabloRun\" tests=\"${run_count}\" failures=\"${err_count}\" disabled=\"0\" skipped=\"0\" errors=\"${err_count}\" >" >> report.xml
for ((i = 0 ; i < ${run_count} ; i++ ));
do
echo "    <testcase name=\"${ini_list[$i]}\">" >> report.xml
echo "      <system-out>[[ATTACHMENT|run_${out_list[$i]}.txt]]</system-out>" >> report.xml
if [ ${success_list[$i]} -eq 0 ]
then
echo "        <failure />" >> report.xml
fi
echo "    </testcase>" >> report.xml
done

echo "  </testsuite>" >> report.xml
echo "</testsuites>" >> report.xml

exit $err_count

