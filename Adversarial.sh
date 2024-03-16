
listData="datasetScientificBackground.csv datasetArtisticBackground.csv"

for Z in $listData
    do

            for X in $(seq 0 6)
                do
                    python Adversarial.py $Z $X
                done

    done

