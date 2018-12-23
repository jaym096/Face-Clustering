from imutils import paths
import face_recognition
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

print "[info] quantifying faces..."
imagePaths = list(paths.list_images(args["dataset"]))
data = []

#loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    #load the input image and convert it from RGB (openCV ordering)
    #to dlib ordering (RGB)
    print("[INFO] processing image {}/{}").format(i+1, len(imagePaths))
    print(imagePath)
    image = cv2.imread(imagePath)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb_img, model=args["detection_method"])
    
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb_img, boxes)
    
    # build a dictionary of the image path, bounding box location,
    # and facial encodings for the current image
    d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
        for (box, enc) in zip(boxes, encodings)]
    data.extend(d)
    
# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()