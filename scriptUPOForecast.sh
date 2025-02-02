#!/bin/bash
w=20
h=1
echo "going to run for window " $w
#for (( i = 2 ; i <= 5 ; i++ ))
for ed in 1 2 3 6 7 8 10 12 14 24 25 32 44
#for ed in 3 6 7 8 10 12 14 24 25 32 44
#for ed in 1 2
do
    echo "Running with arguments" $ed $w $h
    window="w"$w
    horizon="h"$h
	/usr/bin/Rscript --vanilla predictUPOData.R $ed $w $h > resultEd$ed$window$horizon
done
