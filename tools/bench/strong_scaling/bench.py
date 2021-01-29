#!/bin/python

from math import log2
import os
import sys
import subprocess


machine_threads_per_node=40
machine_gpus_per_nodes=4

executable_path="./test_solver"
template_dir=os.path.dirname(os.path.realpath(__file__))

# Replace keys from params in src with values from params and write to dst
def replace_in_file(src_filename, dst_filename, params):
    f_in  = open( src_filename, "rt" )
    data = f_in.read()
    f_in.close()

    for key in params.keys() :
      data = data.replace( key, str(params[key]) )

    os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
    f_out = open( dst_filename, "wt"  )
    f_out.write(data)
    f_out.close()


def run_testcase( problem_size, block_size, amr_frequency, group_size, nb_nodes, mpi_per_node ):
  
  dst_dir = "bench/run_"+str(problem_size)+"_"+str(block_size)+"x"+str(block_size)+"_"+str(nb_nodes)+"nodesx"+str(mpi_per_node)+"mpi_amr"+str(amr_frequency)+"_group"+str(group_size)

  # Create blast.ini
  ini_src = os.path.join(template_dir, "blast_tmpl.ini")
  ini_dst = os.path.join(dst_dir, "blast.ini")

  ini_params = {}
  ini_params["<block_size>"] = block_size
  lmax = int( log2(problem_size/block_size) )
  ini_params["<amr_level_max>"] = lmax
  ini_params["<amr_level_min>"] = lmax-2 
  ini_params["<amr_frequency>"] = amr_frequency
  ini_params["<group_size>"] = group_size

  replace_in_file(ini_src, ini_dst, ini_params)

  # Create job.slurm
  slurm_src = os.path.join(template_dir, "job_tmpl.slurm")
  slurm_dst = os.path.join(dst_dir, "job.slurm")

  slurm_params = {}
  slurm_params["<nb_nodes>"] = nb_nodes
  slurm_params["<nb_mpi>"] = nb_nodes*mpi_per_node
  slurm_params["<nb_gpus>"] = machine_gpus_per_nodes
  slurm_params["<nb_threads>"] = int(machine_threads_per_node/mpi_per_node)

  replace_in_file(slurm_src, slurm_dst, slurm_params)

  # Safely create a simlink to executable in bench dir
  try:
    os.unlink(os.path.join(dst_dir, "test_solver"))
  except FileNotFoundError:
    pass
  os.symlink(os.path.realpath(executable_path), os.path.join(dst_dir, "test_solver"))

  p = subprocess.Popen(["sbatch", "job.slurm"], cwd=dst_dir)
  p.wait()


#run_testcase( problem_size=1024, block_size=8, amr_frequency=1, group_size=2048, nb_nodes=1, mpi_per_node=1 )
run_testcase( problem_size=1024, block_size=8, amr_frequency=1, group_size=2048, nb_nodes=1, mpi_per_node=2 )
#run_testcase( problem_size=1024, block_size=8, amr_frequency=1, group_size=2048, nb_nodes=1, mpi_per_node=4 )
#run_testcase( problem_size=1024, block_size=8, amr_frequency=1, group_size=2048, nb_nodes=1, mpi_per_node=8 )
#run_testcase( problem_size=1024, block_size=8, amr_frequency=1, group_size=2048, nb_nodes=2, mpi_per_node=4 )


    

