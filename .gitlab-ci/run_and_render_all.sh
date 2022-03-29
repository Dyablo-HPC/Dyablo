set +e
set -x

err_count=0
run_count=0

run_and_render(){
    run_count=$((run_count+1))
    ini_file=$1
    pvsm_file=$2
    png_outfile=$3
    .gitlab-ci/run_testcase_mpi.sh $ini_file 4
    if [ $? -eq 0 ]
    then
	    echo "run ${ini_file} success"
        .gitlab-ci/render.sh $pvsm_file
        mv render.png $png_outfile
    else
	    echo "run ${ini_file} fail"
	    err_count=$((err_count+1))
        touch $png_outfile
    fi    
}

run_and_render test_blast_3D_block.ini visu_blast_3D_block.pvsm blast_3d_block.png 
run_and_render test_blast_3D.ini visu_blast_3D_cell.pvsm blast_3d_cell.png

run_and_render test_blast_2D_block.ini visu_blast_2D_block.pvsm blast_2d_block.png 
run_and_render test_blast_2D.ini visu_blast_2D_cell.pvsm blast_2d_cell.png 

run_and_render test_riemann_2D.ini visu_riemann_2D_cell.pvsm riemann_2d_cell.png
run_and_render test_riemann_2D_block.ini visu_riemann_2D_block.pvsm riemann_2d_block.png

run_and_render test_gravity_spheres_3D.ini visu_gravity_spheres_3D.pvsm gravity_3D.png 

echo "${err_count}/${run_count} runs failed"
exit $err_count
