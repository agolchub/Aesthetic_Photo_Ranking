# Aesthetic_Photo_Ranking

## Building a model
The buildmodel.py script will build a keras model. The model's definition is currently coded into the script and changes require a change to the code of the model. To use the model run the buildmodel.py script with the following arguements:
    ./buildmodel.py -o <path to save model> -w <width of input image in pixels> -h <height of imput image in pixels>

## Training a model
To train the model use the train.py script. Example command for training:
    ./train.py -i ./model-3/ -o ./model-3-train1 -t ./databaserelease2/Forests-2/Training/ -v ./databaserelease2/Forests-2/Validation/ -e 100 -b 50 -l 0.001 -d 0.00005