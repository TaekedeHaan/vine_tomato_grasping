# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:04:59 2020

@author: taeke
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:32:31 2020

@author: taeke
"""

## imports ##
import os # os.sep
import cv2

# custom functions
from detect_truss.util import plot_segments
from detect_truss.util import load_rgb
from detect_truss.util import make_dirs

from detect_truss.segment_image import segment_truss

# ls | cat -n | while read n f; do mv "$f" `printf "%03d.png" $n`; done
if __name__ == '__main__':

    N = 50              # tomato file to load
    extension = ".png"
    dataset = "depth_blue" # "tomato_rot" #  
    save = True

    pwd_current = os.path.dirname(__file__)
    pwd_data = os.path.join(pwd_current, "data", dataset)
    pwd_results = os.path.join(pwd_current, "results", dataset, "02_segment")
    
    make_dirs(pwd_results)
    
    count = 0
    
    for i_tomato in range(1, N):
    
        tomato_ID = str(i_tomato).zfill(3)
        tomato_name = tomato_ID
        file_name = tomato_name + extension
        

        img_rgb = load_rgb(pwd_data, file_name, horizontal = True)
    
        # color spaces
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_hue = img_hsv[:, :, 0] # hue

        background, tomato, peduncle = segment_truss(img_hue, 
                                                     save = save, 
                                                     name = tomato_name, 
                                                     pwd = pwd_results) 
        
        # VISUALIZE
        name = tomato_ID + "_img"
        plot_segments(img_rgb, background, tomato, peduncle, 
                      name = name, pwd = pwd_results)
      
        count = count + 1
        print("completed image %d out of %d" %(count, N))
        
        #        truss = cv2.bitwise_or(tomato,peduncle)
#        peduncle_empty = np.zeros(tomato.shape, dtype = np.uint8)
#    
#
#        
#
#        name = tomato_ID + "_img_1"
#        plot_segments(img_rgb, background, truss, peduncle_empty, 
#                                      file_name = name, pwd = pwd_results)   
#    