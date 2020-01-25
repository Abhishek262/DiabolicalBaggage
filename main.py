from alexnet import alexnet_model
from alexnet import freeze_layer


model = alexnet_model(n_classes=2,freeze = [1,1,1,1,1,1,1,1,1])
