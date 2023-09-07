for file in ./model_checkpoint/*
    do
    if test -f $file
    then
        echo $file
    fi
    done
done