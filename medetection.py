import time
import cv2
import numpy as np
import os
import azure.cognitiveservices.speech as speechsdk
import pyaudio
import wave
from pydub import AudioSegment
from pydub.playback import play
import pyaudio

def speack(text):
    speech_config = speechsdk.SpeechConfig(subscription="80145a4fe150471c9284a5acb2e58fe3", region="francecentral")
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name='en-US-JennyNeural'
    audio_config = speechsdk.AudioConfig(filename="test.mp3".format(name=text[:30]))
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Get text from the console and synthesize to the default speaker.


    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")


    time.sleep(1)



    # sound = AudioSegment.from_file("D:/Dokumente/Git/compvision/test.mp3", format="mp3")
    # sound.export("test.wav", format="wav")

    chunk = 1024
    wf = wave.open("test.mp3", "rb")
    p = pyaudio.PyAudio()
    print(wf.getnchannels())
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)

    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()

    p.terminate()






def main():
    # Load the target images of the people you want to recognize
    target_img1 = cv2.imread("images/b.jpg")
    target_img2 = cv2.imread("images/r.jpg")
    target_img3 = cv2.imread("images/b2.jpg")

    # Convert the target images to grayscale
    target_gray1 = cv2.cvtColor(target_img1, cv2.COLOR_BGR2GRAY)
    target_gray2 = cv2.cvtColor(target_img2, cv2.COLOR_BGR2GRAY)
    target_gray3 = cv2.cvtColor(target_img3, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade classifier for face detection
    face_detector = cv2.CascadeClassifier("haarcascade.xml")

    # Detect faces in the target images
    faces1 = face_detector.detectMultiScale(target_gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces2 = face_detector.detectMultiScale(target_gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces3 = face_detector.detectMultiScale(target_gray3, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extract the face regions from the target images
    for (x, y, w, h) in faces1:
        target_face1 = target_gray1[y:y+h, x:x+w]
    for (x, y, w, h) in faces2:
        target_face2 = target_gray2[y:y+h, x:x+w]
    for (x, y, w, h) in faces3:
        target_face3 = target_gray3[y:y+h, x:x+w]

    # Resize the face regions to the standard size (100x100)
    target_face1 = cv2.resize(target_face1, (100, 100))
    target_face2 = cv2.resize(target_face2, (100, 100))
    target_face3 = cv2.resize(target_face3, (100, 100))

    # Initialize the face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the face recognizer on the target faces
    face_recognizer.train([target_face1, target_face2, target_face3], np.array([0, 1, 2]))

    # Open the video capture object
    cap = cv2.VideoCapture(0)
    name = ""
    found = False
    while not found:
        # Read a frame from the video
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Extract the face region from the frame
            face = gray[y:y+h, x:x+w]

            # Resize the face region to the standard size (100x100)
            face = cv2.resize(face, (100, 100))

            # Predict the identity of the face
            label, confidence = face_recognizer.predict(face)

            # Check if the confidence is below a threshold (e.g. 50)
            
            if confidence < 110:
                # If the confidence is low, display the predicted label
                print(confidence)
                if label == 0:
                    name = "Benjamin"
                elif label == 1:
                    name = "Ronja"
                elif label == 2:
                    name = "Benjamin"
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                found = False

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()
    
    #speack("Welcome " + name + " to this house.")



if __name__ == "__main__":
    main()













# import cv2
# import numpy as np

# # Load the reference image
# reference_image_benjamin = cv2.imread("images/b.jpg", 0)
# cv2.resize(reference_image_benjamin, (64, 64))
# reference_image_ronja = cv2.imread("images/r.jpg", 0)

# # Start the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture each frame
#     ret, frame = cap.read()
    
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces in the frame
#     face_cascade = cv2.CascadeClassifier("haarcascade.xml")
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
#     # For each face detected
#     for (x, y, w, h) in faces:

#         # Extract the face area and resize it
#         face_area = gray[y:y+h, x:x+w]
#         face_area = cv2.resize(face_area, (reference_image_benjamin.shape[1], reference_image_benjamin.shape[0]))
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
#         # Compare the face area with the reference image
#         result_b = cv2.matchTemplate(face_area, reference_image_benjamin, cv2.TM_CCOEFF_NORMED)
#         result_r = cv2.matchTemplate(face_area, reference_image_ronja, cv2.TM_CCOEFF_NORMED)
#         # Check if the comparison result is above a certain threshold
#         threshold = 0.4
#         loc_b = np.where(result_b >= threshold)
#         loc_r = np.where(result_r >= threshold)
        
        
        
#         print(loc_b[0])
#         # If it is a match, draw a rectangle around the face and display the result
#         if len(loc_b[0]) > 0:
#             print("Benjamin:" + str(loc_b[0]))
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, "Benjamin", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         elif len(loc_r[0]) > 0:
#             print("Ronja:" + str(loc_r[0]))
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
#             cv2.putText(frame, "Ronja", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    
#     # Display the result
#     cv2.imshow("Result", frame)
    
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and destroy all windows
# cap.release()
# cv2.destroyAllWindows()
