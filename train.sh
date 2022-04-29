#pip install tensorflow-gpu==2.4
#pip install scikit-learn
#pip install scikit-image
nvidia-smi
python ./train.py --special_model2 -o agnet.18.res -f mse -b 20 -e 1000 -l 0.1 -d 0.1 -n -m 0.09 -t ./databaserelease2/MixedDataset/train/ -v ./databaserelease2/MixedDataset/val/
#python ./train.py -i agnet.specialmodel.6.1 -o agnet.specialmodel.6.2 -b 50 -e 2 -l 0.01 -n -t ./databaserelease2/NatureDataset2/train/ -v ./databaserelease2/NatureDataset2/val
#python ./train.py -i agnet.specialmodel.6.2 -o agnet.specialmodel.6.3 -b 50 -e 200 -l 0.001 -d 0.001 -n -t ./databaserelease2/NatureDataset2/train/ -v ./databaserelease2/NatureDataset/val
#python ./train.py -i agnet.simplemodel.3.meadows -o agnet.simplemodel.3.meadows.1 -b 100 -e 100 -l 0.005 -n -d 0.00009 -t ./databaserelease2/NatureDataset2/train2/ -v ./databaserelease2/NatureDataset2/val/
#python ./train.py -i agnet.simplemodel.3.meadows.1 -o agnet.simplemodel.3.meadows.2 -b 100 -e 100 -l 0.01 -n -d 0.0005 -t ./databaserelease2/NatureDataset2/train2/ -v ./databaserelease2/NatureDataset2/val/

