This is the code of CNN which was written by me for one of Hackathon. Along with this we had used occlusion to indicate the tissue in the image affected the most. I took snippets of code for occlusion from 1-2 online articles and modified it. Hence I am not including it in this

Dataset used - BreakHis 

The BreakHis Dataset is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X). It contains 2,480  benign and 5,429 malignant samples . 

The dataset BreaKHis is divided into two main groups: Benign tumors and Malignant tumors. A benign tumor is a tumor that does not invade its surrounding tissue or spread around the body. A malignant tumor is a tumor that may invade its surrounding tissue or spread around the body. Hence benign tumors on rare occasion may actually be life-threatening, and general rule aren't as bad as the malignant tumors. 

The images are divided on the basis of patients . The images of approximately 75% patients (benign and malignant combined) are included in the training data and testing data comprises of rest . 

Tensorflow is used in the backend in Convolutional Neural Network . A testing accuracy of more than 85% is achieved with 5 epochs.
