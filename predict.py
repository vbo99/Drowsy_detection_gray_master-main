from tensorflow import keras
#from keras.models import model_from_json
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
from threading import Thread
from pygame import mixer 
import os
#import datetime
from datetime import datetime,date
#from keras.applications.imagenet_utils import preprocess_input
from imutils.video import WebcamVideoStream
def sound_alarm(path):
    # play an alarm sound
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()

def loadModel(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weight_path)
    #print("Loaded model from disk")
    # evaluate loaded model on test data
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predictImage(img, model):

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    #images=preprocess_input(images)
    classes = model.predict(images)
    return classes


if __name__ == "__main__":
    # Define counter
    COUNTER = 0
    ALARM_ON = True
    MAX_FRAME = 20
    name="The"
    # Load model
    base_dir = os.path.dirname(__file__)
    prototxt_path = os.path.join('trained_model/model_data/deploy.prototxt')
    caffemodel_path = os.path.join('trained_model/model_data/weights.caffemodel')
	# Read the model_detect face
    model_face = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
	
    model = keras.models.load_model( "trained_model/SqueezeNet_NIR.h5")

    camera = WebcamVideoStream(src=0).start()

    # loop over frames from the video stream
    counter = 0
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale channels)
        frame = camera.read()
        frame = imutils.resize(frame, width = 400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        model_face.setInput(blob)
        detections = model_face.forward()
        im = frame.copy()
        #im[:,:,1]=0
        #im[:,:,2]=0
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # loop over the face detections
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            confidence = detections[0, 0, i, 2]
            if (confidence > 0.5):
                roi = gray[startY:endY, startX:endX]
                predict_img=1
                try:
                    shape = cv2.resize(roi,(100, 100))
                    predict_img = predictImage(shape, model=model)
                    print (predict_img)
                except:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
                #cv2.putText(frame, name, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
                    counter += 1
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
                if predict_img<0.5:
                    COUNTER += 1
                    cv2.putText(frame, "Closing", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "NF: {:.2f}".format(COUNTER), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                    if COUNTER >= MAX_FRAME:
					
                        try:
                            date_object = date.today()
                            time_object = datetime.now().time()
                            time_object = time_object.strftime("%H-%M-%S")
                            os.mkdir(str(date_object))
                            cv2.imwrite(str(date_object)+"/" + str(time_object) + ".jpg", frame)
                        except:
                            cv2.imwrite(str(date_object)+"/" + str(time_object) + ".jpg", frame)
                        filename = 'alarm.wav'
                        sound_alarm(filename)

                        # draw an alarm on the frame
                        cv2.putText(frame, "DROWSY!", (100, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    COUNTER = 0
                    #ALARM_ON = False
                    cv2.putText(frame, "Opening", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    camera.stop()

