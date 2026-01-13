# run job

the current directory is `ne_qmmm_project/data`

here for single job:

```sh
# usage 1: single job
python ../scripts/run_reaxff.py -f reaxff_$jobflag  

# usage 2: batch jobs (n=50)
bash job.sh $jobflag
```

currently, jobflag can be `small1` or `system1`


please configure the environment that `../scripts/run_reaxff.py` can work!

```sh
bash collect.sh $jobflag
```
will help you collect a zip file of results.


NOTE:

* `OMP_THREAD_NUM` should be properly set for lammps in CMD ENVs.
