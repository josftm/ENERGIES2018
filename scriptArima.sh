#!/bin/bash
w=15
h=1
#for (( i = 2 ; i <= 5 ; i++ ))
for ed in  2 3 6 7 8 10 12 14 24 25 32 44
#for ed in 3 6 7 8 10 12 14 24 25 32 44
#for ed in 1 2
do
    echo "Running fir Building " $ed 
    for w in 7 10 15 20
    do
        window="w"$w
        horizon="h"$h
	    #/usr/bin/Rscript --vanilla arima.R $ed $w $h > resultsArimaEd$ed$window$horizon
	    /usr/bin/Rscript --vanilla arima.R $ed $w $h 2> /dev/null
    done
done
