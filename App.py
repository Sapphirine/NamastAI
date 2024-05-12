import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase, ClientSettings
import cv2
import numpy as np
import av
import tensorflow as tf
import time
import pandas as pd
import math
import requests
import queue


model_url = "https://raw.githubusercontent.com/balachander-sa/Namestai_streamlit/main/Movenet_lightning_unit8.tflite"
# Download the TFLite model from GitHub
response = requests.get(model_url)
model_content = response.content

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_content=model_content)
interpreter.allocate_tensors()

github_csv_url = "https://raw.githubusercontent.com/balachander-sa/Namestai_streamlit/main/Mean_var.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(github_csv_url)
result_queue: "queue.Queue[float]" = queue.Queue()
color_map = {'A':'green', 'B':'lime', 'C':'yellow', 'D':'orange', 'F': 'red'}

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    keypoint_indx = [5,6,7,8,11,12,13,14]

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

    for idx in keypoint_indx:
        kp = shaped[idx]
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 8, (255,0,0),2)

def return_Keypoints(frame, interpreter):

    # Reshape image
    img = frame.copy()
    img = tf.image.resize(np.expand_dims(img, axis=0), [192,192])
    input_image = tf.cast(img, dtype=tf.uint8)

    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints_final = keypoints_with_scores[0,0].T[0:2]
    draw_keypoints(frame, keypoints_with_scores, 0.01)

    return frame, keypoints_final

def calculate_angle(a,b,c):

    AB = np.array(b) - np.array(a)
    BC = np.array(c) - np.array(b)

    dot_product = np.dot(AB, BC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)
    
    denom = magnitude_AB * magnitude_BC
    if denom == 0:
        denom = np.nan
    
    cos_theta = dot_product / (denom)
    theta = np.arccos(cos_theta)

    # Convert to degrees
    angle_degrees = np.degrees(theta)

    return angle_degrees 


def get_angles(frame):
    op_image, kp = return_Keypoints(frame, interpreter)

    LSA = calculate_angle(kp[:,0],kp[:,5],kp[:,7])
    RSA = calculate_angle(kp[:,0],kp[:,6],kp[:,8])
    LEA = calculate_angle(kp[:,5],kp[:,7],kp[:,9])
    REA = calculate_angle(kp[:,6],kp[:,8],kp[:,10])
    LHA = calculate_angle(kp[:,5],kp[:,11],kp[:,13])
    RHA = calculate_angle(kp[:,6],kp[:,12],kp[:,14])
    LKA = calculate_angle(kp[:,11],kp[:,13],kp[:,15])
    RKA = calculate_angle(kp[:,12],kp[:,14],kp[:,16])
    Angles = [LSA, RSA, LEA, REA, LHA, RHA, LKA, RKA]

    return op_image, Angles
    
# Function to calculate z-scores
def calculate_z_score(value, mean, std_dev):
    return (value - mean) / std_dev

# Function to assign grades based on z-scores
def assign_grade(z_score):
    if z_score <= 1:
        grade =  "A"
    elif (z_score > 1 and z_score <=2):
        grade = "B"
    elif (z_score > 2 and z_score <=3):
        grade = "C"
    elif (z_score > 3 and z_score <=4):
        grade = "D"
    else:
        grade = "F"

    return grade
    
def apply_colors(row):
    color = color_map.get(row['Grade'], 'white')  # Default color is white if grade is not found
    return ['background-color: {}'.format(color)] * len(row)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    op_frame, angles = get_angles(image)
    
    result_queue.put(angles)
    return av.VideoFrame.from_ndarray(op_frame, format="bgr24")

st.title("NamestAI: A yoga fitness app by Balachander")

# Set client settings for WebRTC
client_settings = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
# Render the video stream 



frame_rate = st.sidebar.slider("Select the refresh speed", 10, 30, 15)
col1, col2 = st.columns([1, 1])
with col1:
    webrtc_ctx = webrtc_streamer(
        key="NamestAI",
        video_frame_callback=video_frame_callback,
        client_settings=client_settings,
        media_stream_constraints={
        "video": {"frameRate": {"ideal": frame_rate}},
    },
        video_html_attrs={
            "style": {"width": "100%", "height": "100%", "margin": "left"},
            "controls": False,
            "autoPlay": True,
        },
    )
Final_result = []


with col2:
    if webrtc_ctx.state.playing:
        angle_placeholder = st.empty()
        correction = st.empty()
        while True:
            result = result_queue.get()
            Final_result.append(result)
            row_names = ["Left shoulder angle", "RIght shoulder angle", "Left elbow angle", "Right elbow angle", "Left hip angle", "Right hip angle", "Left knee angle", "Right knee angle"]
            i = 0
            LSA_mean = df.loc[i, 'LSA_mean']
            RSA_mean = df.loc[i, 'RSA_mean'] 
            LEA_mean = df.loc[i, 'LEA_mean']
            REA_mean = df.loc[i, 'REA_mean']
            LHA_mean = df.loc[i, 'LHA_mean']
            RHA_mean = df.loc[i, 'RHA_mean']
            LKA_mean = df.loc[i, 'LKA_mean']
            RKA_mean = df.loc[i, 'RKA_mean']
            LSA_std_dev = df.loc[i, 'LSA_std_dev']
            RSA_std_dev = df.loc[i, 'RSA_std_dev']
            LEA_std_dev = df.loc[i, 'LEA_std_dev']
            REA_std_dev = df.loc[i, 'REA_std_dev']
            LHA_std_dev = df.loc[i, 'LHA_std_dev']
            RHA_std_dev = df.loc[i, 'RHA_std_dev']
            LKA_std_dev = df.loc[i, 'LKA_std_dev']
            RKA_std_dev = df.loc[i, 'RKA_std_dev']

            LSA_val = (result[0] - LSA_mean) / LSA_std_dev
            RSA_val = (result[1] - RSA_mean) / RSA_std_dev 
            LEA_val = (result[2] - LEA_mean) / LEA_std_dev 
            REA_val = (result[3] - REA_mean) / REA_std_dev  
            LHA_val = (result[4] - LHA_mean) / LHA_std_dev
            RHA_val = (result[5] - RHA_mean) / RHA_std_dev 
            LKA_val = (result[6] - LKA_mean) / LKA_std_dev 
            RKA_val = (result[7] - RKA_mean) / RKA_std_dev             
            
            angle_vals = [LSA_val, RSA_val, LEA_val, REA_val, LHA_val, RHA_val, LKA_val, RKA_val]
            grades = []
            angle_val = [ '%.2f' % elem for elem in angle_vals ]
            for angs in angle_vals:                 
                grades.append(assign_grade(np.abs(angs)))

            angle_df = pd.DataFrame({"Angle": angle_val, "Grade": grades}, index=row_names)
            # Function to apply color to the entire row

            angle_df = angle_df.style.apply(apply_colors, axis=1)

            styled_table = angle_df.set_table_styles([
            {"selector": "th", "props": [("font-size", "10px")]},  # Target row names
            {"selector": "td", "props": [("font-size", "10px")]}   # Target data cells
        ])
            
            angle_placeholder.dataframe(styled_table)

            
            max_index = np.argmax(np.abs(angle_vals))
            worst_joint = row_names[max_index]

            action = "extend" if np.sign(max_index) >= 0 else "flex"
            op_text = str("Please ")+ action + str(" your ")+ worst_joint
            correction.text(op_text)
            time.sleep(1/frame_rate)