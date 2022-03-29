# face recognition app using streamlit; incoprating face mesh technology and algorithms
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import face_recognition
import pandas as pd
import os

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from numpy import asarray

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo2.jpg'
# CONSTANTS
PATH_DATA = 'data/DB.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name', 'description']
COLS_ENCODE = [f'v{i}' for i in range(128)]


def init_data(data_path=PATH_DATA):
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)

# convert image from opened file to np.array


def byte_to_array(image_in_byte):
    return cv2.imdecode(
        np.frombuffer(image_in_byte.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

# convert opencv BRG to regular RGB mode


def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

# convert face distance to similirity likelyhood


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))




st.title("DynoSci powered by Streamlit")

st.markdown(
    """
    <style>
    [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
    width: 350px
    margin-left: -350px
    
    
   
    
    </style>
    
    
    """,
    unsafe_allow_html=True,

)

st.sidebar.button('Logout')

st.sidebar.title('DynoSci Facemesh')
st.sidebar.subheader('Parameters')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # code to resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


app_mode = st.sidebar.selectbox('Choose App mode',
                                ['About Site', 'Run on Image', 'Run on Video', 'Run Face Recognition'])

if app_mode == 'About Site':
    st.markdown('DynoSci is an artificial intelligence platform that incorporates neural '
                'networks solutions and models to offer excellent face recognition algorithms.'
                + ' The project is developed in Python as its website language '
                  'in scientific computing and OpenCv libraries for'
                  ' face recognition methods. Streamlit is a framework implemented for faster building of highly aesthetic machine'
                  ' learning apps GUI')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
        width: 350px
        }
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
        width: 350px
        margin-left: -350px
        }
    
    
        </style>
    
    
        """,
        unsafe_allow_html=True,

    )
    st.image('pythonlogo.png')
    st.image('opencvlogo.png')

    st.markdown('''
            # The Dev Team \n
            We are excited to develop this project as part of demonstrating numerical methods and analysis to build a facemesh and face identification application. The team has worked exceptionally hard to provide you with a glimpse of a plethora of possibilities with machine learning applied to security biometrics in our day to day lives. 
            Our team of developers is working hard to build and add more exciting features to project DynoSci.
            
        ''')

elif app_mode == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
        width: 350px
        }
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
        width: 350px
        margin-left: -350px
        }


        </style>


        """,
        unsafe_allow_html=True,

    )

    st.markdown("Detected Faces")
    kpi1_text = st.markdown("0")

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        image: ndarray = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    face_count = 0

    # dashboard code

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        # Face Landmarks code
        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec)

            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
            st.subheader('Output Image')
            st.image(out_image, use_column_width=True)

elif app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # max faces
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    with kpi4:
        st.markdown("**Face ID**")
        kpi4_text = st.markdown("alpha feature")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_faces=max_faces) as face_mesh:
        prevTime = 0

        while vid.isOpened():
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if record:
                # st.checkbox("Recording", value=True)
                out.write(frame)
            # Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()

elif app_mode == 'Run Face Recognition':
    choice = st.selectbox("Select Option", [
        "Face Detection",
        "Face Detection 2",
        "Face Verification"
    ])

    fig = plt.figure()
    if choice == "Face Detection":
        # load the image
        uploaded_file = st.file_uploader("Choose File", type=["jpg", "png"])
        if uploaded_file is not None:
            data = asarray(Image.open(uploaded_file))
            # plot the image
            plt.axis("off")
            plt.imshow(data)
            # get the context for drawing boxes
            ax = plt.gca()
            # plot each box
            # load image from file
            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            faces = detector.detect_faces(data)
            for face in faces:
                # get coordinates
                x, y, width, height = face['box']
                # create the shape
                rect = Rectangle((x, y), width, height, fill=False, color='maroon')
                # draw the box
                ax.add_patch(rect)
                # draw the dots
                for _, value in face['keypoints'].items():
                    # create and draw dot
                    dot = Circle(value, radius=2, color='maroon')
                    ax.add_patch(dot)
            # show the plot
            st.pyplot(fig)

    elif choice == "Face Detection 2":
        uploaded_file = st.file_uploader("Choose File", type=["jpg", "png"])
        if uploaded_file is not None:
            column1, column2 = st.columns(2)
            image = Image.open(uploaded_file)
            with column1:
                size = 450, 450
                resized_image = image.thumbnail(size)
                image.save("thumb.png")
                st.image("thumb.png")
            pixels = asarray(image)
            plt.axis("off")
            plt.imshow(pixels)
            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            results = detector.detect_faces(pixels)
            # extract the bounding box from the first face
            x1, y1, width, height = results[0]["box"]
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize((224, 224))  # Rodgers -> You can just save this as image
            face_array = asarray(image)
            with column2:
                plt.imshow(face_array)
                st.pyplot(fig)

    elif choice == "Face Verification":
        image_byte = st.file_uploader(
            label="Select a picture contains faces:", type=['jpg', 'png']
        )
        # detect faces in the loaded image
        max_faces = 0
        rois = []  # region of interests (arrays of face areas)
        if image_byte is not None:
            image_array = byte_to_array(image_byte)
            face_locations = face_recognition.face_locations(image_array)
            for idx, (top, right, bottom, left) in enumerate(face_locations):
                # save face region of interest to list
                rois.append(image_array[top:bottom, left:right].copy())

                # Draw a box around the face and lable it
                cv2.rectangle(image_array, (left, top),
                              (right, bottom), COLOR_DARK, 2)
                cv2.rectangle(
                    image_array, (left, bottom + 35),
                    (right, bottom), COLOR_DARK, cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    image_array, f"#{idx}", (left + 5, bottom + 25),
                    font, .55, COLOR_WHITE, 1
                )

            st.image(BGR_to_RGB(image_array), width=720)
            max_faces = len(face_locations)

        if max_faces > 0:
            # select interested face in picture
            face_idx = st.selectbox("Select face#", range(max_faces))
            roi = rois[face_idx]
            st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

            # initial database for known faces
            DB = init_data()
            face_encodings = DB[COLS_ENCODE].values
            dataframe = DB[COLS_INFO]

            # compare roi to known faces, show distances and similarities
            face_to_compare = face_recognition.face_encodings(roi)[0]
            dataframe['distance'] = face_recognition.face_distance(
                face_encodings, face_to_compare
            )
            dataframe['similarity'] = dataframe.distance.apply(
                lambda distance: f"{face_distance_to_conf(distance):0.2%}"
            )
            st.dataframe(
                dataframe.sort_values("distance").iloc[:5]
                    .set_index('name')
            )

            # add roi to known database
            if st.checkbox('add it to knonwn faces'):
                face_name = st.text_input('Name:', '')
                face_des = st.text_input('Desciption:', '')
                if st.button('add'):
                    encoding = face_to_compare.tolist()
                    DB.loc[len(DB)] = [face_name, face_des] + encoding
                    DB.to_csv(PATH_DATA, index=False)
        else:
            st.write('No human face detected.')
