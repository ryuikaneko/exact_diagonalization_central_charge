#!/bin/bash

prog=../src/no_eigvec.py
#prog=../src/no_eigvec_no_matrix.py

for L in \
`seq 3 28`
do

for momk in \
`seq 0 $((L/2))`
do

date

echo ${L} ${momk}
python ${prog} -L ${L} -momk ${momk} > dat_L${L}_momk${momk}

date

done

done
