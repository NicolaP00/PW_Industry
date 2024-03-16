# PW_Industry

- Run mainScript.py (for each result) as follows:
python FoolingAI.py fileCSV targetN loss

fileCSV = datasetArtisticBackground.csv | datasetScientificBackground.csv

targetN = 0 | 1 | 2 | 3 | 4 | 5 | 6

loss = no | lf | af | adv

- Run Adversarial.py (for each result) as follows:
python FoolingAI.py fileCSV targetN

fileCSV = datasetArtisticBackground.csv | datasetScientificBackground.csv

targetN = 0 | 1 | 2 | 3 | 4 | 5 | 6

- Run FoolingAI.sh (for all the results) as follows:
./FoolingAI.sh

- Run Adversarial.sh (for all the results) as follows:
./Adversarial.sh
