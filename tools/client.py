# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:07:55 2019

@author: ShivamRatnakar, Ashish Rao
"""

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import requests
import json
import cv2
import base64
from PIL import Image
import io
import numpy as np
#headers = {'content-type': 'application/json'}

url = 'http://127.0.0.1:3066'


depth_image = cv2.imread('/home/kartvat3/openVino/instance_segmentation/data/rgbanddepth/depth_images/1525694104_depth.png', cv2.CV_16UC1)
retval, buffer = cv2.imencode('.png', depth_image)
jpg_as_text_di = base64.b64encode(buffer)

image = cv2.imread('/home/kartvat3/openVino/instance_segmentation/data/rgbanddepth/rgb_images/1525694104_rgb.png')
retval, buffer = cv2.imencode('.png', image)
jpg_as_text = base64.b64encode(buffer)
params = {'depth_image' : jpg_as_text_di, 'image': jpg_as_text}


for i in range(5):
    r = requests.post(url, params=params)
    input("hi")

print(r.text)
