## Project Files for NamastAI 

**Angles_pos1.csv** contains the angles of all the joints of the 9 YouTubers referenced. Further processed to obtain the joint angles.

**App.py** The deployable app. The app which is deployed in Streamlit. It process the live video stream and outputs the results like grades and comments in a streamlit dashboard.

**Calculate_mean_var.ipynb** Once the Angles are returned their standard deviation and variance needs to be obtained. This notebook processes this information.

**Data_preprocessing.ipynb** After Datapreprocessing, the calculation of joint angles is implemented from keypoints detected from the model is available here. 

**Mean_var.csv** Contains all the necessary joint angle's means and standard deviation needed for grading. 

**Movenet_lightning_unit8.tflite** Contains Tensorflow's Movenet lightning model. 

**NamastAI-Report.pdf** is the final report of this project.

**requirements.txt** contains the necessary libraries used. 


For latest updates on this project please check https://github.com/balachander-sa/Namestai_streamlit.
