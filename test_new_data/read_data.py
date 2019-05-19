#########
import os
import shutil

    #lines = f.read().split("\n")
    #del(lines[len(lines)-1])
folders= os.listdir('C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade')  
training_folder=('C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/allData')
#print('FOLDERS',folders)
for i in folders:
	files_1=os.listdir('C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i + '/'+'001')
	files_2=os.listdir('C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i + '/'+'002')
	files_3=os.listdir('C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i + '/'+'003')
	files_4=os.listdir('C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i + '/'+'004')
	files_5=os.listdir('C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i + '/'+'005')
	files_6=os.listdir('C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i + '/'+'006')

	for file in files_1:
		if not os.path.isfile(file):
			shutil.move( 'C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i +'/001/'+file, training_folder)
		else:
			os.remove(file)
	for file in files_2:
		if not os.path.isfile(file):
			shutil.move( 'C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i +'/002/'+file, training_folder)
		else:
			os.remove(file)
	for file in files_3:
		if not os.path.isfile(file):
			shutil.move( 'C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i +'/003/'+file, training_folder)
		else:
			os.remove(file)
	for file in files_4:
		if not os.path.isfile(file):
			shutil.move( 'C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i +'/004/'+file, training_folder)
		else:
			os.remove(file)
	for file in files_5:
		if not os.path.isfile(file):
			shutil.move( 'C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i +'/005/'+file, training_folder)
		else:
			os.remove(file)
	for file in files_6:
		if not os.path.isfile(file):
			shutil.move( 'C:/Users/1/Desktop/andro/Final Semester Tartuu/prepare/emotion_recognition/test_new_data/CK/cohn-kanade/' + i +'/006/'+file, training_folder)
		else:
			os.remove(file)	
"""
#############
def getlabels(source1,source2=None): #each image_name contains the label as a last symbol. This function takes the destination
    
    #--inputs--
    #source1,source2- path to the image folders
    #--returns--
    #extracts labels from the image name
    
    labels=[]
    files= os.listdir(source1)  
    files2=os.listdir(source2)  
    for file in files:   
        file_name=os.path.splitext(os.path.basename(file))[0]
        label=int(file_name.split('_')[4][5])
        labels.append(label) 
    if source2!=None:
        for file in files2:   
            file_name=os.path.splitext(os.path.basename(file))[0]
            label=int(file_name.split('_')[4][5])
            labels.append(label)
    labels=np.array(labels)
    return labels
"""