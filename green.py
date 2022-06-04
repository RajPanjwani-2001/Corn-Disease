import cv2
import glob
import numpy as np
import pickle

fromPath = 'C:/Users/Raj/MyProg/Corn_Disease/CornDataset/'
className = {0:'Blight',1:'Common_Rust',2:'Gray_Leaf_Spot',3:'Healthy'}

li = []
labels =[]
for key,values in className.items() :
    for img in glob.glob(fromPath + values + '/*.jpg'):
        cv_img = cv2.imread(img)
        print(img)
        b, g, r = cv2.split(cv_img)
        # cv2.imshow('image',g)
        g = cv2.resize(g,(224,224))
        g = g.reshape(g.shape[0], g.shape[1], 1)
        print('g: ',g.shape)
        li.append(g)
        labels.append(key)

toPath = 'C:/Users/Raj/MyProg/Corn_Disease/codes/'
features = toPath+'Combined_2'+".pkl"
class_labels = toPath+'ClassLabels_2'+".pkl"

li = np.array(li)
labels = np.array(labels)

fo = open(features, "wb")
pickle.dump(li, fo)
fo.close()

fo = open(class_labels, "wb")
pickle.dump(labels, fo)
fo.close()

print(labels)
print(li.shape)
