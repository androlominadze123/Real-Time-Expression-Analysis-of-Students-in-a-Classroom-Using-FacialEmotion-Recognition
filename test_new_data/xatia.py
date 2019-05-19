#########
import os
import shutil
with open('cases_train.txt') as f:
    lines = f.read().split("\n")
    del(lines[len(lines)-1])
    training_folder='/home/ubuntu/preprocessing/maincode/files/training/'
    for i in lines:
            files_0=os.listdir('/home/ubuntu/preprocessing/dataset/' +i+'/0')
            files_1=os.listdir('/home/ubuntu/preprocessing/dataset/' +i+'/1')
            for file in files_0:
                shutil.move( '/home/ubuntu/preprocessing/dataset/' +i+'/0/'+file, training_folder)
            for file in files_1:
                shutil.move( '/home/ubuntu/preprocessing/dataset/' +i+'/1/'+file, training_folder)
#############
def getlabels(source1,source2=None): #each image_name contains the label as a last symbol. This function takes the destination
    """
    --inputs--
    source1,path to the image folders
    --returns--
    extracts labels from the image name
    """
    labels=[]
    files= os.listdir(source1)  
    for file in files:   
        file_name=os.path.splitext(os.path.basename(file))[0]
        label=int(file_name.split('_')[1][2])
		if label==0:
			label=...
	    elif label==1
        labels.append(label) 
    return labels
######
def getnpfeatures(source1):
    """
    --inputs--
    source1, path to the image folders
    --returns--
    returns np.array of images
    """
    print('npfeaturescalled')
    data=[]
    for myFile in glob.glob (source1):   
        image = load_img(myFile, target_size=(48, 48))
        image = img_to_array(image) 
        data.append(image)
    data=np.array(data)
    return data    