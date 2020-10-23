import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import random

from torchvision import transforms
import torchvision.transforms.functional as TF


class Rain():

    def __init__(self, rain_intensity=1.0, haze_intensity=4.0, gamma=2.0, light_correction=0.9):

        self.rain_intensity = rain_intensity
        self.haze_intensity = haze_intensity
        self.gamma = gamma
        self.light_correction = light_correction
        self.frame_id = 1

        self.cap = cv2.VideoCapture(r'./rain_streaks/videoplayback.mp4')

    def _get_rain_layer(self):

        ret, frame = self.cap.read()
        if ret:
            rain_layer = frame
        else: # if reach the last frame, read from the begining
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            rain_layer = frame

        rain_layer = cv2.cvtColor(rain_layer, cv2.COLOR_BGR2RGB) / 255.0
        rain_layer = np.array(rain_layer, dtype=np.float32)

        return rain_layer


    def _create_haze_layer(self, rain_layer):
        return 0.1*np.ones_like(rain_layer)

    def forward(self, img):

        # get input image size
        h, w, c = img.shape

        # create a rain layer
        rain_layer = self._get_rain_layer()

        rain_layer = cv2.resize(rain_layer, (w, h))
        rain_layer = cv2.blur(rain_layer, (3, 3))
        rain_layer = rain_layer * (1 - cv2.boxFilter(img, -1, (int(w/10), int(h/10))))

        # create a haze layer
        haze_layer = self._create_haze_layer(rain_layer)

        # combine the rain layer and haze layer together
        rain_layer = self.rain_intensity*rain_layer + \
                     self.haze_intensity*haze_layer

        # synthesize an output image (screen blend)
        img_out = 1 - (1 - rain_layer) * (1 - img)

        # gamma and light correction
        img_out = self.light_correction*(img_out**self.gamma)

        # check boundary
        img_out = np.clip(img_out, a_min=0, a_max=1.)

        return img_out

