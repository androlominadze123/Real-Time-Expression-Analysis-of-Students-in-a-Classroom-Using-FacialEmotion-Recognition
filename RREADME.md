# Real-Time-Expression-Analysis-of-Students-in-a-Classroom-Using-FacialEmotion-Recognition
Master's Thesis


The success of service robotics highly depends on a smooth Robot-User Interaction. Thus, a robot should be able to extract information
just from the face of its user, e.g. must identify the emotional state. Human accuracy for classifying an image of a facein one of
7 different emotions is 65 percents.  One can observe the difficulty ofthis task by trying to classify the FER2013 dataset 
images manually. Despite theseproblems, robot platforms give the possibility to cope with these struggles with computertechniques.

The FER2013 dataset can be downloaded from here:
https://drive.google.com/file/d/1RqUE6POqY_TMo-2KT7v3f0A17g041cCo/view?usp=sharing

ER2013 is an opensource dataset; it was first created by Pierre-Luc Carrierand Aaron Courvilleet al.[26]. 
The dataset contains 35.887 grayscale, 48x48 sized faceimages (9). These images are labelled with seven different emotions.

•  4593 images- Angry
•  547 images- Disgust
•  5121 images- Fear
•  8989 images- Happy
•  6077 images- Sad
•  4002 images- Surprise
•  6198 images- Neutral

The dataset contains three main columns:
  •Emotioncolumn contains the numeric label of the facial expression.
  •Pixelscolumn contains the pixel values of the individual images. It represents thematrix of pixel values of the image.
  •Usagecolumn contains the purposes of each data sample. There are three differentlabels in this column: 
   Training, PublicTest, and PrivateTest. For training purposes,there are 28,709 data samples. 
   The public test set consist of 3,589 data samples,and the private test set consists of another 3,589 data samples.


After you download the repository you need to run in command prompt the following script:
python emo_rec.py

To test the model for other datasets, you can download some example datasets from here:
https://drive.google.com/file/d/1v2YbUVFnG8hLP0FxwvsHE8s6mH1tKJKy/view?usp=sharing

After you download the archieved file you need to extract it in the same directory and then you can test the model by running
the Jupyter notebooks:
testkdef.ipynb
testjaffe.ipynb

