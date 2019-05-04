from PIL import ImageTk
from PIL import Image as ImG
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from playsound import playsound
import os
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time
import numpy as np
from decimal import Decimal
import splash as spl

test = ""
test_files = []
testimage = ""
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
# face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
emoji_faces = []
for index, emotion in enumerate(EMOTIONS):
    emoji_faces.append(cv2.imread('emojis/' + emotion.lower() + '.png', -1))
faceCascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')


def askfolder(a):
    global test
    test = filedialog.askdirectory()
    folentry.insert(0, test)
    global test_files
    test_files = [f for f in os.listdir(test) if os.path.isfile(os.path.join(test, f))]
    if "desktop.ini" in test_files:
        test_files.remove("desktop.ini")

def main(a):
    def realtime(a):
        result = np.array((1, 7))
        video_capture = cv2.VideoCapture(0)
        video_capture.set(3, 640)  # WIDTH
        video_capture.set(4, 480)  # HEIGHT
        once = False

        # save current time
        prev_time = time.time()

        # start webcam feed
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            # mirror the frame
            frame = cv2.flip(frame, 1, 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # find face in the frame
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                # required region for the face
                # roi_color = frame[y-90:y+h+70, x-50:x+w+50]

                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # save the detected face
                # cv2.imwrite(save_loc, roi)
                # draw a rectangle bounding the face
                cv2.rectangle(frame, (x - 10, y - 70),
                              (x + w + 20, y + h + 40), (15, 175, 61), 4)

                # keeps track of waiting time for emotion recognition
                curr_time = time.time()
                # do prediction only when the required elapsed time has passed
                if curr_time - prev_time >= 1:
                    # read the saved image
                    # img = cv2.imread(save_loc, 0)

                    if roi is not None:
                        # indicates that prediction has been done atleast once
                        once = True

                        # resize image for the model
                        result = emotion_classifier.predict(roi)
                        emotion_probability = np.max(result[0])
                        label = EMOTIONS[result[0].argmax()]
                        emoji_index = result[0].argmax()
                        print(label)

                    # save the time when the last face recognition task was done
                    prev_time = time.time()

                if once == True:
                    total_sum = np.sum(result[0])
                    # select the emoji face with highest confidence
                    emoji_face = emoji_faces[emoji_index]
                    for index, emotion in enumerate(EMOTIONS):
                        text = str(
                            round(Decimal(result[0][index] / total_sum * 100), 2)) + "%"
                        # for drawing progress bar
                        cv2.rectangle(frame, (100, index * 20 + 10),
                                      (100 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                                      (255, 0, 0), -1)
                        # for putting emotion labels
                        cv2.putText(frame, emotion, (10, index * 20 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
                        # for putting percentage confidence
                        cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # overlay emoji on the frame for all the channels
                    for c in range(0, 3):
                        # for doing overlay we need to assign weights to both foreground and background
                        foreground = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0)
                        background = frame[350:470, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
                        frame[350:470, 10:130, c] = foreground + background
                break

            # Display the resulting frame
            cv2.imshow('Realtime Emotion Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()



    def imagefile(a,b,c):
        global current_test
        testimage = tkvar.get()
        testimage = test + "/" + testimage
        print(testimage)
        result = np.array((1, 7))

        frame = cv2.imread(testimage)
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find face in the frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # required region for the face
            # roi_color = frame[y-90:y+h+70, x-50:x+w+50]

            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # save the detected face
            # cv2.imwrite(save_loc, roi)
            # draw a rectangle bounding the face
            cv2.rectangle(frame, (x - 10, y - 70),
                          (x + w + 20, y + h + 40), (15, 175, 61), 4)

            if roi is not None:
                # resize image for the model
                result = emotion_classifier.predict(roi)
                emotion_probability = np.max(result[0])
                label = EMOTIONS[result[0].argmax()]
                emoji_index = result[0].argmax()
                print(label)

                file = "media/audio/"+label+".mp3"
                playsound(file)


            # save the time when the last face recognition task was done
            prev_time = time.time()


            total_sum = np.sum(result[0])
            # select the emoji face with highest confidence
            emoji_face = emoji_faces[emoji_index]
            for index, emotion in enumerate(EMOTIONS):
                text = str(
                    round(Decimal(result[0][index] / total_sum * 100), 2)) + "%"
                # for drawing progress bar
                cv2.rectangle(frame, (100, index * 20 + 10),
                              (100 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                              (255, 0, 0), -1)
                # for putting emotion labels
                cv2.putText(frame, emotion, (10, index * 20 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
                # for putting percentage confidence
                cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # overlay emoji on the frame for all the channels
            try:
                for c in range(0, 3):
                    # for doing overlay we need to assign weights to both foreground and background
                    foreground = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0)
                    background = frame[350:470, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
                    frame[350:470, 10:130, c] = foreground + background
            except:
                pass
            break

        # Display the resulting frame
        cv2.imshow('Output', frame)
        cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def exitf(a):
        root.destroy()




    fol.destroy()
    root = Tk()

    root.iconbitmap('media/icons/main.ico')
    tkvar = StringVar(root)
    root.title("Human Emotion Recognizer")
    drop_down_menu = OptionMenu(root, tkvar, *test_files)
    Label(root, text="To recognize a single file, Choose from the dropdown: ").grid(row=0, column=0, sticky="w")
    img = ImageTk.PhotoImage(ImG.open("media/images/ddown.jpg"))
    panel = Label(root, image=img).grid(row=1, column=0, rowspan=2, sticky="ew")
    drop_down_menu.grid(row=1, column=1, sticky="ew")
    tkvar.trace('w', imagefile)
    ttk.Separator(root).grid(row=4, pady=2, padx=2, columnspan=2, sticky="ew")
    ttk.Separator(root).grid(row=5, pady=2, padx=2, columnspan=2, sticky="ew")
    Label(root, text="For realtime emotion detection, Click Realtime:").grid(row=5, column=0, sticky=W)
    Label(root, text="Press Q to close the Camera Window:").grid(row=6, column=0, sticky=W)
    img2 = ImageTk.PhotoImage(ImG.open("media/images/realtime.jpg"))
    panel2 = Label(root, image=img2).grid(row=7, column=1, rowspan=2, sticky="nes")
    rt = Button(root, text="Realtime")
    rt.bind("<Button-1>", realtime)
    rt.grid(row=7, column=0, rowspan = 2, sticky="ew")
    ex = Button(root, text="Exit")
    ex.bind("<Button-1>", exitf)
    ex.grid(row=9, columnspan=2, sticky="ew")

    root.mainloop()

spl.splash()
fol = Tk()

fol.iconbitmap('media/icons/save.ico')
fol.title("Testing Folder Selection")
Label(fol, text="Choose the folder containing Testing Images:").grid(row=0, column=0, sticky=W)
img = ImageTk.PhotoImage(ImG.open("media/images/folder_selection.jpg").resize((250, 250), ImG.ANTIALIAS))
panel = Label(fol, image=img).grid(row=1, column=0, rowspan=2, sticky="ew")
folentry = Entry(fol, width=77)
folentry.grid(row=3, sticky=W, column=0)
ch = Button(fol, text="Browse")
ch.bind("<Button-1>", askfolder)
ch.grid(row=3, column=1, sticky=E)
nx = Button(fol, text="Next")
nx.bind("<Button-1>", main)
nx.grid(row=4, columnspan=2, sticky="ew")

fol.mainloop()