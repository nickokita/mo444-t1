time for ((i=0;i<8192;i++)) ; do
    id=$(( $i % 4 ))
    echo "id-$id = $i"
    python3 compute-diamonds-normal-equation.py ../diamonds-dataset/diamonds-train.csv ../diamonds-dataset/diamonds-test.csv $i | tee -a out-$id.txt &
    if (( $i % 4 )) ; then
        wait
    fi
done
