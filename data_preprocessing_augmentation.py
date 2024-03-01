# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:07:00 2023

@author: Mobeen  
"""

#%%
import os
import cv2
from glob import glob
import imageio
import numpy as np
from albumentations import HorizontalFlip, VerticalFlip, Rotate,RandomBrightness,RandomBrightnessContrast,MedianBlur,ElasticTransform,GridDistortion,MotionBlur,OpticalDistortion

def load_data(path):
    #train_x = sorted(glob(os.path.join(path,"training","images","*.tif")))
    train_x = sorted(glob(os.path.join(path,"train","images","*.jpg")))
    #train_y = sorted(glob(os.path.join(path,"training","1st_manual","*.gif")))
    train_y = sorted(glob(os.path.join(path,"train","mask","*.png")))
    
    #test_x = sorted(glob(os.path.join(path,"test","images","*.tif")))
    #test_y = sorted(glob(os.path.join(path,"test","1st_manual","*.gif")))
    
    test_x = sorted(glob(os.path.join(path,"test","images","*.jpg")))
    test_y = sorted(glob(os.path.join(path,"test","mask","*.png")))
    
    return train_x,train_y, test_x, test_y


if __name__ == "__main__":
    
    np.random.seed(42)
    
    #data_path = r"C:\Users\Ashu2\Downloads\archive0\DRIVE"
    data_path = r"C:\Users\Ashu2\Downloads\CHASEDB1\chase"
    #data_path = r"C:\Users\Ashu2\Downloads\stare"
   
    [train_x,train_y, test_x, test_y] = load_data(data_path)
    
    print ( len(train_x) , len(test_x) ,len(train_y) , len(test_y))
    
    
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
            
       # drive --> 1.1 gamma, clip size =2.0 , no norm here, unet add drop out layers 0.1 ,0.1,0.4, preporcess of y labels 
       # chase --> 1.2 gamma, clip size =3.0, same for stare,  no norm here, unet add drop out layers 0.1 ,0.1,0.4, preporcess of y labels 
    
    create_dir("newdatac1a/train/image")
    create_dir("newdatac1a/train/mask")
    create_dir("newdatac1a/test/image")
    create_dir("newdatac1a/test/mask")
    
    #create_dir("C:/Users/Ashu2/Desktop/retina_vessels/chase1/train/image/")
    #create_dir("C:/Users/Ashu2/Desktop/retina_vessels/chase1/train/mask/")     
    #create_dir("C:/Users/Ashu2/Desktop/retina_vessels/chase1/test/image/")      
    #create_dir("C:/Users/Ashu2/Desktop/retina_vessels/chase1/test/mask/")      
    
#%%

            
#%%

                #import cv2
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import median_filter
    import numpy as np


 

    #data augmentation
    from tqdm import tqdm
  
    
    def gammaCorrection(src, gamma=1.2):
        invGamma = 1 / gamma

        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)

        return cv2.LUT(src, table)
        
    def augment_data(image,mask,save_path,augment=True):
            
        #H = 192
        #W = 192
        
        print(save_path)
                    
        for idx, (x,y) in tqdm(enumerate(zip(image,mask)),total=len(image)):
        #extracting names of the image
            name = x.split("\\")[-1].split(".")[0]
            print(name)
            
            name1 = y.split("\\")[-1].split(".")[0]
            print(name1)
        #Reading image and mask
            x = cv2.imread(x,cv2.IMREAD_COLOR)
            y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            #y = imageio.mimread(y)[0] #, activate for .gif format images
            print(x.shape)
            print(y.shape)
            
            
            
            if augment == True:
                #pass
                #x=rgb2gray(x)
                #y=rgb2gray(y)
                #assert np.max(y)==1
            
                #assert np.max(y)==1
                # laplacian image gradients
                #x = cv2.Laplacian(gray,cv2.CV_64F)
                gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                
                
                    #x=cv2.equalizeHist(gray)
                # iamge restoration usijng denoising colored
                    #   x = cv2.fastNlMeansDenoisingColored(x,None,10,10,7,21)
                #x = cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #x = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                
                #gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                
                
                # laplacian image gradients
                #x = cv2.Laplacian(gray,cv2.CV_64F)
                
                # Create the sharpening kernel
                #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  
                # Sharpen the image
                #x = cv2.filter2D(x, -1, kernel)
  
                # Remove noise using a Gaussian filter
                #x = cv2.GaussianBlur(x, (7, 7), 0)
                
                
                #gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                x = clahe.apply(gray)
                
                #x = cv2.normalize(x, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #x=cv2.medianBlur(x, 3)
                x = gammaCorrection(x, 1.2)
               # x = gammaCorrection(x, 1.2)
                
                #x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) 
                #x=cv2.equalizeHist(x)
                
                #kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]
                    #]
                    #)
                #x = cv2.filter2D(x,-1,kernel)
                
               # x= cv2.medianBlur(x, 5) 
                
                #graya = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
                clahea = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                ya = clahea.apply(y)
                
                #ya=cv2.medianBlur(ya, 3)
                
                y = gammaCorrection(ya, 1.2)
                
                # probability of horizantal flip applied to an image
                aug = RandomBrightness(p=1)#HorizontalFlip(p=1.0)
                augmented = aug(image=x,mask=y)
                x1 = augmented['image']
                y1 = augmented['mask']
                
                aug = MotionBlur(p=1)#VerticalFlip(p=1.0)
                augmented = aug(image=x,mask=y)
                x2 = augmented['image']
                y2 = augmented['mask']
                
                      #aug = ElasticTransform(p=1.0,alpha=120,sigma=120*0.05)
                      #augmented = aug(image=x,mask=y)
                      #x3 = augmented['image']
                     # y3 = augmented['mask']
                     
                aug = GridDistortion(p=1)#Rotate(limit=45,p=1.0)
                augmented = aug(image=x,mask=y)
                x3 = augmented['image']
                y3 = augmented['mask']
                
                aug = OpticalDistortion(p=1)#Rotate(limit=45,p=1.0)
                augmented = aug(image=x,mask=y)
                x4 = augmented['image']
                y4 = augmented['mask']
                
                aug = Rotate(limit=35, p=1)#Rotate(limit=45,p=1.0)
                augmented = aug(image=x,mask=y)
                x5 = augmented['image']
                y5 = augmented['mask']
                
                aug = VerticalFlip(p=1)#Rotate(limit=45,p=1.0)
                augmented = aug(image=x,mask=y)
                x6 = augmented['image']
                y6 = augmented['mask']
                
                aug = HorizontalFlip(p=1)#Rotate(limit=45,p=1.0)
                augmented = aug(image=x,mask=y)
                x7 = augmented['image']
                y7 = augmented['mask']
                
                aug = ElasticTransform(p=1)#Rotate(limit=45,p=1.0)
                augmented = aug(image=x,mask=y)
                x8 = augmented['image']
                y8 = augmented['mask']
                
                aug = MedianBlur(blur_limit=3, p=1)#Rotate(limit=45,p=1.0)
                augmented = aug(image=x,mask=y)
                x9 = augmented['image']
                y9 = augmented['mask']
                
                aug = RandomBrightnessContrast(p=1)#Rotate(limit=45,p=1.0)
                augmented = aug(image=x,mask=y)
                x10 = augmented['image']
                y10 = augmented['mask']
                
                X = [x,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
                Y = [y,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10]
              
            
            else:
                
                # laplacian image gradients
                #x1 = cv2.Laplacian(x,cv2.CV_64F)
                gray1 = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                
               # x1=cv2.equalizeHist(gray1)
               # x1 = cv2.fastNlMeansDenoising(x1,None,10,10,7,21)
                #x = cv2.normalize(x, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                #gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                
                
                # laplacian image gradients
               # x1 = cv2.Laplacian(x1,cv2.CV_64F)
                
                # Create the sharpening kernel
                #kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  
                # Sharpen the image
                #x1 = cv2.filter2D(x1, -1, kernel1)
  
                # Remove noise using a Gaussian filter
                #x1 = cv2.GaussianBlur(x1, (7, 7), 0)
                
                #gray1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
                
                
                #x1=cv2.equalizeHist(gray1)
                #x1 = gammaCorrection(x1, 1.2)
                
                clahe1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                x1 = clahe1.apply(gray1)
                #x1=cv2.medianBlur(x1, 3)
                #x1 = cv2.normalize(x1, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #x1 = gammaCorrection(x1, 1.2)
                x1 = gammaCorrection(x1, 1.2)
                

                #x1 = cv2.cvtColor(x1, cv2.COLOR_GRAY2RGB) 
                            # Median filtering
               # kernel1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]
                     # ]
                     # )
                #x1 = cv2.filter2D(x1,-1,kernel1)
                
                #x1= cv2.medianBlur(x1, 5) 
                            
                            #x1 = cv2.normalize(gray1, None, 0, 255, cv2.NORM_MINMAX)
                              
             
                
                
                #gray2 = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
                clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                y1 = clahe2.apply(y)
                #y1=cv2.medianBlur(y1, 3)
                
                
                
                y1 = gammaCorrection(y1, 1.2)
                
                X = [x1]
                Y = [y1]
                
            index = 0
            
            for i,m in zip(X,Y):
                
                #i = cv2.resize(i,(W,H))
                #m = cv2.resize(m,(W,H))
       
                #if len(X) == 1:
                    
                tmp_image_name = f"{name}_{index}.jpg"
                tmp_mask_name = f"{name1}_{index}.jpg"
                
                #else:
                   # tmp_image_name = f"{name}_{index}.jpg"
                    #tmp_mask_name = f"{name}_{index}.jpg"
                image_path = os.path.join(save_path,"image",tmp_image_name)
                mask_path = os.path.join(save_path,"mask",tmp_mask_name)
                    #print(image_path)
                   # print(mask_path)
                cv2.imwrite(image_path, i)
                cv2.imwrite(mask_path, m)
                index+=1
                
    
#%%

    
    
 #%%   
    
    augment_data(train_x,train_y,"newdatac1a//train/",augment=True)      
    augment_data(test_x,test_y,"newdatac1a//test/",augment=False)
    
    
  #%%







