#!/usr/bin/env python
import cv2
import numpy as np
import os.path

dir = '/media/root/My Passport/new-data/'
dir2 = dir + 'images/'
l=[]
files=dict()
names=[]

with open(dir+'labels.csv', 'r') as f:
    l=f.readlines()

for i in l:
    name,value = i.split(',')
    value=value.rstrip()
    if(value == '1'):
        filename = 'gun.' + name + '.jpg'
    else:
        filename = 'nogun.' + name + '.jpg'

    name = name + '.jpg'
    files[name] = filename

width = 400
height = 400
dim = (width,height)

for i in files.keys():
    if(os.path.isfile(dir2 + str(i))):
        img = cv2.imread(dir2 + str(i), cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(dir + 'processed-images/' + files[i],resized)
