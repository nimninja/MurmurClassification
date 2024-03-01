from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import csv
import wave
import contextlib
from PYMONGOTEST import collection_name
from MONGODBGET import get_database
import base64
from PIL import Image
from pandas import DataFrame

dbname = get_database()


def tflow(output_filepath, fname, patient_name, diagnosis):

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_Model_real.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(output_filepath).convert("RGB")

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

    if "murmur" in output_filepath:
        check = "Murmur Prese..."
    else:
        check = "Murmur Absen..."

    if (classaccurate == check):
        accurate = True
    else:
        accurate = False

    print(accurate)

    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print(duration)

    confidence_score = confidence_score * 100

    # assume data contains your decoded image


    # Convert the image to base64 format
    with open(output_filepath, "rb") as e:
        encoded_image = base64.b64encode(e.read())
        encoded_image = str(encoded_image)
        encoded_image = encoded_image.replace("'", "")
        encoded_image = encoded_image.replace(encoded_image[0], "", 1)

    if option == 1:
        diagnosis_1 = {
            "patient_name" : patient_name,
            "prediction" : classaccurate,
            "confidence" : confidence_score,
            "audio_length" : duration,
            "img" : encoded_image
        }
        collection_name.insert_one(diagnosis_1)

    if option == 2:
        diagnosis_2 = {
            "patient_name": patient_name,
            "diagnosis": diagnosis,
            "confidence": confidence_score,
            "audio_length": duration,
            "img": encoded_image
        }

        collection_name.insert_one(diagnosis_2)


    #with open("CCSV", "a", newline='') as csv_file:
        #writer = csv.writer(csv_file)
        #writer.writerow([output_filepath, duration, confidence_score, classacurate, accurate])


def convert_wav_to_spectrogram(input_folder, output_folder, patient_name, diagnosis):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all WAV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

            # Load the audio file
            y, sr = librosa.load(input_filepath)

            # Generate the spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

            # Plot and save the spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram of {}'.format(filename))
            plt.savefig(output_filepath)
            plt.close()

            print('Spectrogram saved for {}'.format(filename))
            fname = input_filepath
            tflow(output_filepath, fname, patient_name, diagnosis)
            print(output_filepath)


def option1():
    input_folder = input("Enter path to your folder of wave files: ")
    output_folder = input("Enter output location: ")
    patient_name = input("Enter name: ")
    convert_wav_to_spectrogram(input_folder, output_folder, patient_name, diagnosis = "")


def option2():
    input_folder = input("Enter path to your folder of wave files: ")
    output_folder = input("Enter output location: ")
    patient_name = input("Enter name: ")
    diagnosis = input("Enter diagnosis: ")
    convert_wav_to_spectrogram(input_folder, output_folder, patient_name, diagnosis)


def option3():
    dbname = get_database()
    patient_name = input("patient_name: ")
    collection_name = dbname["diagnosis_info"]
    item_details = collection_name.find({"patient_name" : patient_name})
    for item in item_details:
        # convert the dictionary objects to dataframe
        items_df = DataFrame(item_details)

        # see the magic
        print(items_df)


option = int(input("option: "))
if option == 1:
    option1()

if option == 2:
    option2()

if option == 3:
    option3()

else:
    print("enter 1 or 2 or 3 again")
    exit()

