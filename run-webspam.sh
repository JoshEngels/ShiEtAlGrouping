#!/bin/bash
# Look, the chance that any of these finish at the same time is negligable, so
# just write em all to the same file
for groups in 20000 40000
do
  for reps in 2 4
  do 
    for batchsize in 10000 80000
    do
      for numbatches in 1 2 4
      do
        for dimension in 200 400
        do
          taskset -c 0-100 ./build/run ../Webspam/data 10000 ../Webspam/indices 100 1024 $groups $reps $batchsize $numbatches $dimension >> webspam.out &
          sleep 240
        done
      done
    done
  done
done

wait
