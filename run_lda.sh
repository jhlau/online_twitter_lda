#/bin/bash

#parameters
num_topics=10

#globals
dates=`ls input/ | cut -f 1-1 -d \. | sort | uniq`
first=1

prev=""
for date in $dates
do
    echo "processing datetime = $date (previous datetime = $prev)"
    if [ $first -eq 1 ]
    then
        #run the first one without online
        out_dir=`echo $date | cut -f 2-2 -d-`
        time python lda.py -f input/$date.text -t input/$date.time -o output-$out_dir -k $num_topics
        prev=$out_dir
    else
        time python lda.py -f input/$date.text -t input/$date.time \
            -m output-$prev/model.dat -o output-$date
        prev=$date
    fi
    
    first=0
done
