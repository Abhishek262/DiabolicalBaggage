#!/usr/bin/env python
# coding: utf-8

# In[60]:


import cv2
import numpy as np


# In[61]:


j = 1
l = []
l2 = []


# In[62]:


width = 224
height = 224
dim = (width,height)


# In[63]:


with open('/root/docs/DiabolicalBaggage/Data/guns/a.txt','r') as a:
    l = a.readlines()
    for i in l:
        x = i.rstrip('\n')
        l2.append(x)


# In[64]:


for i in l2:
    img = cv2.imread('/root/docs/DiabolicalBaggage/Data/guns/'+str(i),cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite('/root/docs/DiabolicalBaggage/Data/guns2/'+str(j)+".png",resized)
    j += 1


# In[ ]:




