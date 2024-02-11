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

def test(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all WAV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            fname = input_filepath
            tflow(output_filepath, fname)


def tflow(output_filepath, fname):

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
    classacurate = class_name[2:].strip()

    if "murmur" in output_filepath:
        check = "Murmur Prese..."
    else:
        check = "Murmur Absen..."

    if (classacurate == check):
        accurate = True
    else:
        accurate = False

    print(accurate)

    #with contextlib.closing(wave.open(fname, 'r')) as f:
        #frames = f.getnframes()
        #rate = f.getframerate()
        #duration = frames / float(rate)
        #print(duration)


    #with open("CCSV", "a", newline='') as csv_file:
        #writer = csv.writer(csv_file)
        #writer.writerow([output_filepath, duration, confidence_score, classacurate, accurate])



def convert_wav_to_spectrogram(input_folder, output_folder):
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
            tflow(output_filepath, fname)




def option1():
    input_folder = input("Enter path to your input .wav file: ")
    output_folder = input("Enter output location: ")
    convert_wav_to_spectrogram(input_folder, output_folder)

option1()


