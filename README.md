# DiabolicalBaggage: Automated baggage scanning using ML


## Datasets:
Two datasets were experimented with. 
* [SIXray](https://github.com/MeioJane/SIXray)
* [GDXray](https://domingomery.ing.puc.cl/material/gdxray/)

## References:
https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/     
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

## Execution:
* Model.py: Sets up the model with the mobilenet image classification architecuture.
* Training.py: Trains the model, with specified dataset.

## Parts
* Firearm classifier
* Other objects such as knives, laptops, shurikens, razor blades,etc

## Final Dataset
*Firearm classifier : Cropped out images of guns vs empty bag space

## To-do 
Perform image segmentation over the dataset to produce actual results
