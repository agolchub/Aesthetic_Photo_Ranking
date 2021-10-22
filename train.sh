pip install tensorflow-gpu==2.6
pip install scikit-learn
pip install scikit-image

python ./train.py --resnet50 -o resnet.model -b 50 -e 50 -r -d 0.0001 -l 0.001 -t ./databaserelease2/NatureDataset/train/ -v ./databaserelease2/NatureDataset/val/
