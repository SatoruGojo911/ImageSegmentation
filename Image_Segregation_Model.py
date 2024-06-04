# -------- IMPORTS -------
import streamlit as st
import pandas as pd
import numpy as np 
import cv2 
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
st.title('Image Segmentation Using Kmeans')
st.header('Upload an Image ')
imag = st.file_uploader(label=' ')
if (imag):
    if imag is None:
        print("Error: Could not load image")
    else:
        file_bytes = np.asarray(bytearray(imag.read()), dtype=np.uint8)# Getting the image data from inputted file 
        image = cv2.imdecode(file_bytes,1) # Decoding the image so that we can further work on the image 
        im =cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # converting the standard BGR to RGB Colouring so that colout is matched properlt
        
        st.image(im) # Displays the origional image 
        st.header('Enter the number of Dominant colors ')
        a =  st.text_input(" ")
        if (a):
            a = int(a) # The no of cluster centres or the dominant colours to extract from the image
            origional_shape = im.shape
            all_pixel_values = im.reshape((-1,3))# Reducing the size of image such that it can be passed through KMeans

            KMS = KMeans(n_clusters = a)
            KMS.fit(all_pixel_values)

            centres = KMS.cluster_centers_
            # The cluster centers are the values of colours which are most dominant colours 

            def plot_colors(a):# This Code helps us plot the most dominant colours 
                img = []
                st.header('The {} Dominant Colors extracted are '.format(a))
                # Plotting these colours as a 100 X 100 block in a continous line 
                for each_col in centres:
                    a = np.zeros((100,100,3),dtype = 'uint8')
                    a[:,:,:] = each_col
                    img.append(a)
                    
                st.image(img) 
            plot_colors(a)
            
            def see_segregated_image(): # This code seperated the image into the dominant colours 
                st.header('The segregated image is ')
                
                new_image = np.zeros((KMS.labels_.shape[0], 3),dtype = 'uint8')
                # Need to place all values from origional colour and change them to the new dominant extracted colours 
                for ix in range(KMS.labels_.shape[0]-1):
                    # If there is a shift in colour scale then this code helps us draw a black line of 2px
                    if (centres[KMS.labels_[ix]].all() != centres[KMS.labels_[ix+1]].all()):
                        new_image[ix] = [0,0,0]
                        new_image[ix+1] = [0,0,0]
                    new_image[ix] = centres[KMS.labels_[ix]]# Placing the dominant colour where its label is present 
                new_image = new_image.reshape(origional_shape[0],origional_shape[1],3) # Changing shape such that making it becomes easier 
                st.image(new_image) # A new dominant image is shown 
            see_segregated_image()# DIsplays the new image 
            
