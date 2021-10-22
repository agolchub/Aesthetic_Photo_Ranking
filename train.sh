pip install tensorflow-gpu==2.6
pip install scikit-learn
pip install scikit-image

python ./train.py --special_model -o model2.train -b 5 -e 1 -r -d 0.0001 -l 0.001 -t ./databaserelease2/NatureDataset/train/ -v ./databaserelease2/NatureDataset/val/
