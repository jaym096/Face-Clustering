# Face-Clustering
Applied Unsupervised learning on the image data-set of various soccer players to generate cluster montage of each individual player.

### STEPS TO RUN THE CODE:
	1. The Data folder contains the Dataset with images of different soccer players. You can add more images according to your 	  convenience.
	2. First Encode all the images and generate pickle file.
           example : python encode_face.py --dataset dataset --encodings encodings.pickle
	3. Once the pickle file is generated generate the cluster montage by running below mentioned code.
           example: python cluster_faces.py --encodings encodings.pickle
