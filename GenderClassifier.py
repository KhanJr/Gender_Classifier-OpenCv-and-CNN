import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file


class FaceCV(object):

    CASE_PATH = "./pretrained_models/haarcascade_frontalface_alt.xml"            # Harcascade model to detect face
    WRN_WEIGHTS_PATH = "./pretrained_models/weights.18-4.06.hdf5"                # These weights are trained on thousands of images to classify gender


    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)


    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=1):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (0, 0, 0), cv2.FILLED)      # Drawing the rectangle on the image_frame
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)       # Offering labeling to the image_frame


    def crop_face(self, imgarray, section, margin=40, size=64):

        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section                                        # x = x-coordinate, y = y-coordinate, w = width_image_frame, h = height_image_frame
        margin = int(min(w,h) * margin / 100)                         # add some margin to the face detected area to include a full head
        x_a = x - margin                                              # Selecting the minimum possible margin to detect only the face
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin

        if x_a < 0:                                                   # The logic here is to traverse threw whole image to detect the face 
            x_b = min(x_b - x_a, img_w-1)                             # Starting points are y - margin, x - margin->
            x_a = 0     
                                                                      # (the value after deleting the margin value to start from left most and upper most point possible)
        if y_a < 0:                                                   # End point the maximum right most and downside point possible.
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0

        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w

        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
            
        cropped = imgarray[y_a: y_b, x_a: x_b]                                                      # Selecting the detected region only and cropped it. 
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)               # Storing image in multidimensional type (point form)
        resized_img = np.array(resized_img)                                                         # Returning the size of the face detected

        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)


    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)   
        os.system('clear')                                    # Using the cascade classifier to detect face
        print("---------------********-------------\n"+
                "Please choose one of the following:\n---------------********-------------\n1. PHOTO FACE CLASSIFICATION\n"+
                "2. VIDEO FACE CLASSIFICATION\n3. LIVE CAMERA CLASSIFICATION\n---------------********-------------")

        temp = int(input())

        if temp == 1:
            print("Enter the image address: ")       # Using the multiple method to classify the image as male or female
            image_path = str(input())                      # Image classifier
            if os.path.exists(image_path) == False:
                print("Error, Invalid address! Please Check the address.")
                exit(0)
            else:
                video_capture = cv2.imread(image_path)
        elif temp == 2:
            print("Enter the video address: ")              # Video classifier
            video_path = str(input())               
            if os.path.exists(video_path) == False:
                print("Error, Invalid address! Please Check the address.")
                exit(0)
            else: 
                video_capture = cv2.VideoCapture(video_path)
        elif temp == 3:                                     # Live cam. classifier
            video_capture = cv2.VideoCapture(0)             # 0 means the default video capture device in OS, can use 1  if using external capture device
        
        while True:                                         # Continuously running facecam until terminated by user, infinite loop, break by key ESC                                                                        

            if temp == 2 or temp == 3:
                if not video_capture.isOpened():
                    sleep(5)   
                ret, frame = video_capture.read()  
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            elif temp == 1:
                frame = video_capture
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(self.face_size, self.face_size)
            )


            if faces is not ():

                face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))                # placeholder for cropped faces
                for i, face in enumerate(faces):
                    face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)  # The larger the frame size the better the classification
                    (x, y, w, h) = cropped
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)
                    face_imgs[i,:,:,:] = face_img
               

                if len(face_imgs) > 0:
                    
                    results = self.model.predict(face_imgs)                  # predict genders of the detected faces
                    predicted_genders = results[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = results[1].dot(ages).flatten()
                   
                                                                             # draw results
                for i, face in enumerate(faces):
                    label = "{}".format("FEMALE" if predicted_genders[i][0] > 0.5 else "MALE")   # setting threshold value to choose between male and female
                    if label == "FEMALE":
                        print(0)
                    elif label == "MALE": 
                        print(1)
                    
                    self.draw_label(frame, (face[0], face[1]), label)
            else:
                print('NO FACE !')

            cv2.imshow('DETECTED FACE', frame)                              # Printing the name on image_frame
            if cv2.waitKey(5) == 27:                                        # ESC key press to close the image_frame
                break
        if temp == 2 or temp == 3:
            video_capture.release()                                         # When everything is done, release the capture
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description="This model classify the gender male of Human",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args

def main():
    os.system('clear')
    args = get_args()
    depth = args.depth
    width = args.width

    face = FaceCV(depth=depth, width=width)

    face.detect_face()


if __name__ == "__main__":
    main()
