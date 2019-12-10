#!/bin/bash

#prog=../src/no_eigvec.py
prog=../src/no_eigvec_no_matrix.py

for L in \
34
#8 10 12 14 16 18 20 22 24 26 28 30 32
#6 ## too small for scipy.sparse.linalg.eigsh
do

for momk in \
`seq 0 $((L/2))`
do

date

echo ${L} ${momk}
python2.7 ${prog} -L ${L} -momk ${momk} > dat_L${L}_momk${momk}

date

done

done
