class config:
    folderPath = 'C:/Users/Raj/MyProg/Corn_Disease/CornDataset/'
    metaDataFile = 'metadata.csv'
    image_size = (128, 128)  # Width, Height
    folders = ['Blight/', 'Common_Rust/', 'Gray_Leaf_Spot/','Healthy/']
    cls_dict = {folders[0]: 0, folders[1]: 1, folders[2]: 2, folders[3]: 3}
