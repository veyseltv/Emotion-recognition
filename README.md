# İmage-Proccessing and Emotion Analysis
 İmage Proccessing with OpenCV

# ABSTRACT

Emotion detection has become one of the most important elements to be considered in all projects related to Affective Computing. Due to the almost endless applications of this new discipline, the development of emotion-sensing technologies has emerged as a highly profitable opportunity for technology companies. In recent years, many new organizations have emerged that are almost exclusively devoted to a particular emotion perception technology. To this end, I use a model of artificial neural networks, where I can read and recognize emotions. I also present the research of all technologies that can be applied for this project. This thesis study enables us to identify and understand the strengths and deficiencies of existing technology for emotion detection. We conclude our thesis by emphasizing the parts that require more research and development and are missing.

# Goal Of Project 

•	This project aims to analyze and determine 7 basic emotions using CNN (convolution neural network). These feelings; it was determined as happy, sad, angry, neutral, disgust, fear and astonishment. To increase the accuracy rate of 14.29% obtained in previous studies to a better point.

# Face Detection

•	Facial recognition is the first step of the project. Haar feature-based classifiers under the name of the OpenCV library were used for face recognition and face detection. This perception is an effective object detection method proposed by Paul Viola and Michael Jones from the article "Rapid Object Detection using a Boosted Cascade of Simple Features".

# Design of Project 

•	Image standardization: Image standardization: Includes various subprocesses such as removing noise in the image, making all images the same in size, and converting from RGB (Red, Green, and Blue) to grayscale. This makes image data available for image analysis.

•	Face detection: This section helps the received data to detect the face. It removes all unnecessary and unwanted backgrounds (except the face) from the picture and only the target reveals the selected face. Also, various methods such as face partitioning techniques and curvature properties are used here. Some algorithms reused in this step include edge detection filters such as Sobel, Prewitt, Laplacian, and canny.

•	Facial component detection: In this part, whatever part of our project will help us determine that region. These areas include eyes, nose, mouth, etc. can. First, identify and track an intense facial point. Then it is necessary as it helps to minimize errors that may arise due to the face turning or aligning.

•	Decision function: After the feature point tracking of the face using parameters such as localized feature Lucas Kanade Optical ﬂow tracker, it is the decision function responsible for detecting the acquired emotion of the subject. These functions make use of classiﬁers such as AdaBoost and SVM for facial emotion recognition. We also benefited from these functions in terms of lightening our project.


# Output Of Project
The purpose of the outputs of this project is to uncover basic recognition and analysis of basic human emotions.

https://github.com/veyseltv/Image-Proccessing/issues/2#issue-704321719




