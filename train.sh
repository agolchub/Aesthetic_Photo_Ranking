#pip install tensorflow-gpu==2.4
#pip install scikit-learn
#pip install scikit-image
nvidia-smi
python ./train.py --special_model -o sptest.model -b 5 -e 50 -r -d 0.0001 -l 0.001 -t ./databaserelease2/NatureDataset/train/ -v ./databaserelease2/NatureDataset/val/
