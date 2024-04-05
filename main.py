import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps, Image
from keras.models import load_model
from MONGODBGET import get_database
from os import environ
from pandas import DataFrame
import random
import base64
import wave
import contextlib
from PYMONGOTEST import collection_name



st.markdown(
    """
<style>
button.st-emotion-cache-nbt3vv.ef3psqc13 {
    height: 100px !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important; 
    color: rgb(255, 225, 255) !important;
    border: 1px solid rgb(255, 255, 255) !important;
    font-size: 100px !important;
}
div.st-emotion-cache-3ps0xc.e1nzilvr5 {
    width: 700px !important;
    font-size: 1000px !important;  
    font-family: "Source Sans", sans-serif;
}
div.st-emotion-cache-3ps0xc e1nzilvr5 {
    --rem: 16;
    text-size-adjust: 100%;
    -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
    -webkit-font-smoothing: auto;
    color-scheme: dark;
    text-transform: none;
    font-weight: 400;
    line-height: 1.6;
    user-select: none;
    cursor: pointer;
    color: rgb(255, 225, 255) !important;
    box-sizing: border-box;
    width: 700px !important;
    font-size: 1000px !important;
    font-family: "Source Sans Pro", sans-serif;
}
div.row-widget.stButton {
    color: rgb(200,200,200) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def tflow(output_path, fname, patient_name):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(r"C:\Users\nimay\PycharmProjects\Heart_Sound_Classification\keras_model_real.h5", compile=False)

    # Load the labels
    class_names = open(r"C:\Users\nimay\PycharmProjects\Heart_Sound_Classification\labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(output_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    classaccurate = class_name[2:].strip()
    if classaccurate == "Murmur Prese...":
        classaccurate = "Murmur Present"
    elif classaccurate == "Murmur Absen...":
        classaccurate = "Murmur  Absent"

    with open(output_path, "rb") as e:
        encoded_image = base64.b64encode(e.read())
        encoded_image = str(encoded_image)
        encoded_image = encoded_image.replace("'", "")
        encoded_image = encoded_image.replace(encoded_image[0], "", 1)
    confidence_score = confidence_score * 100

    #with contextlib.closing(wave.open(fname, 'r')) as f:
        #frames = f.getnframes()
        #rate = f.getframerate()
        #duration = frames / float(rate)
        #print(duration)

    confidence_score = round(confidence_score, 2)
    st.header("Prediction: " + classaccurate)
    st.header("Confidence score is " + str(confidence_score) + "%")
    diagnosis_1 = {
        "patient_name" : patient_name,
        "prediction" : classaccurate,
        "confidence" : confidence_score,
        #"audio_length" : duration,
        "img" : encoded_image
    }
    collection_name.insert_one(diagnosis_1)


def save_spectrogram(wav_file, output_file, patient_name):
    # Load audio file
    y, sr = librosa.load(wav_file)

    # Compute spectrogram
    D = np.abs(librosa.stft(y))

    # Display spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram for {patient_name} : {wav_file.name}')
    plt.tight_layout()

    # Save the spectrogram to the specified location
    plt.savefig(output_file)
    st.pyplot(plt)
    plt.close()
    tflow(output_file, wav_file, patient_name)


if 'button1' not in st.session_state:
    st.session_state.button1 = False


if 'button2' not in st.session_state:
    st.session_state.button2 = False


def click1():
    st.session_state.button2 = False
    st.session_state.button1 = True


def click2():
    st.session_state.button1 = False
    st.session_state.button2 = True


option_1 = st.button(r"$\textsf{\Huge Receive Diagnosis}$", type = "primary", on_click=click1)
option_2 = st.button(r"$\textsf{\Large Patient History}$", key = "primary", on_click=click2)

while st.session_state.button1:
    st.session_state.button2 = False
    audio_path = st.file_uploader("Select the Audio Recording", type="wav")
    patient_name = st.text_input("Enter Patient's Name")

    export_path = environ.get("TEMP")

    export_path = export_path + "\\" + patient_name + "spectrogram.png"

    if audio_path != "" and export_path != "" and patient_name != "":
        save_spectrogram(audio_path, export_path, patient_name)


while st.session_state.button2:
    st.session_state.button1 = False
    dbname = get_database()
    patient_name = st.text_input("Patient Name", key = random.random)
    collection_name = dbname["diagnosis_info"]
    item_details = collection_name.find({"patient_name" : patient_name})
    if item_details == "":
        st.write("No History For This Patient \n Verify Name is Valid")
    for item in item_details:
        # convert the dictionary objects to dataframe
        items_df = DataFrame(item_details)

        # see the magic
        print(items_df)
        st.write(items_df)

