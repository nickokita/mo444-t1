time for ((i=0;i<1023;i++)) ; do
    id=$(( $i % 4 ))
    python3 compute-diamonds-normal-equation.py diamonds-dataset/diamonds-train.csv diamonds-dataset/diamonds-test.csv $i | tee -a out-$id.txt &
    if (( $i % 4 )) ; then
        wait
    fi
done
