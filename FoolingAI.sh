
listLoss="no lf af"

listData="datasetScientificBackground.csv datasetArtisticBackground.csv"

for Z in $listData
    do
        for Y in $listLoss
            do
                for X in $(seq 0 6)
                    do
                        python FoolingAI.py $Z $X $Y
                    done
            done
    done

