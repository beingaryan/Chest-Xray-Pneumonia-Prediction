
<h1 align="center">Chest-Xray-Pneumonia-Prediction</h1>

<div align= "center">
  <h4>Chest-Xray-Pneumonia-Prediction built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts. </h4>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/issues)
[![Forks](https://img.shields.io/github/forks/beingaryan/Chest-Xray-Pneumonia-Predictionsvg?logo=github)](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/network/members)
[![Stargazers](https://img.shields.io/github/stars/beingaryan/Chest-Xray-Pneumonia-Prediction.svg?logo=github)](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/stargazers)
[![Issues](https://img.shields.io/github/issues/beingaryan/Chest-Xray-Pneumonia-Prediction.svg?logo=github)](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/issues)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/aryan-gupta-6a9201191/)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<p align="center"><img src="https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/blob/main/Outputs/Non%20pneumonia.png" width="700" height="400"></p>
<p align="center"><b>Non Pneumonia</b></p>

## :point_down: Support me here!
<a href="https://www.buymeacoffee.com/beingaryan" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
</br>
## :innocent: Motivation
A Deep Learning based model used for the prediction whether a person is suffering from pneumonia or not. The project is absed upon the Standard __Convolutional Based Neural Network__ Architectural Implementation. It incorporates the use of CNN layers with Hyper-parameters tuning. The motivation behind the project is to effectively classify the reports of chest xrays to classify into pneumonia or non-pneumonia cases.
</br></br> The model has been trained for 12 epoch. The model gave an accuracy of __98.51__ on validation sets.




<!---Unable to communicate verbally is a disability. In order to communicate there are many ways, one of the most popular methods is the use of predefined sign languages. The purpose of this project is to bridge the __research gap__ and to contribute to recognize __American sign languages(ASL)__ with maximum efficiency. This repository focuses on the recognition of ASL in real time, converting predicted characters to sentences and output is generated in terms of voice formats. The system is trained by convolutional neural networks for the classification of __26 alphabets__ and one extra alphabet for null character. The proposed work has achieved an efficiency of __99.88%__ on the test set.--->


<p align="center"><img src="https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/blob/main/Outputs/pneumonia.png" width="700" height="400"></p>

<p align="center"><b>Case of Pneumonia</b></p>

## :⚠: TechStack/framework used

- [OpenCV](https://opencv.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)


## :file_folder: Data Distribution
The dataset used can be downloaded here - [Click to Download](https://drive.google.com/drive/folders/1f6QGKHQ2rJD3jCrumAKXPbUg_Q8TenFz?usp=sharing)

This dataset consists of __images__ belonging to 2 classes:
<!---*	__Training Set: 12845 images__<br />
<p align="center"><img src="https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/Analysis/train_data_distribution.png" ></br><b>Train Data Statistics</b></p>
<!---<br />![](Analysis/train_data_distribution.png)<br />--->

<!---*	__Test Set: 4368 images__<br />
<p align="center"><img src="https://github.com/beingaryan/Sign-To-Speech-Conversion/blob/master/Analysis/test_data_Distribution.png" ></br><b>Test Data Statistics</b></p>
<!---<br />![](Analysis/train_data_distribution.png)<br />--->



## :star: Features
Our model is capable of predicting Pneumonia from chest x-ray images with high efficiency. These __predicted images__ are converted to grayscale version for predictions.</br></br>
The model is efficient, since we used a compact __CNN-based architecture__, it’s also computationally efficient and thus making it easier to deploy the model to servers.
<!---
## 🎨 Feature Extraction
* Gaussian filter is used as a pre-processing technique to make the image smooth and eliminate all the irrelevat noise.
* Intensity is analyzed and Non-Maximum suppression is implemented to remove false edges.
* For a better pre-processed image data, double thresholding is implemented to consider only the strong edges in the images.
* All the weak edges are finally removed and only the strong edges are consdered for the further phases. <br />
<br />![](Analysis/fe.png)<br />
The above figure shows pre-processed image with extracted features which is sent to the model for classification.--->


## :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/blob/main/requirements.txt)

## 🚀&nbsp; Installation
1. Start and fork the repository.

2. Clone the repo
```
$ git clone https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction.git
```

3. Change your directory to the cloned repo and create a Python virtual environment named 'test'
```
$ mkvirtualenv test
```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

## :bulb: Working

1. Open terminal. Go into the cloned project directory and type the following command:
```
$ python3 jupyter
```

2. To train the model, open the [Pneumonia_Prediction](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/blob/main/PNEUMONIA_DETECTION.ipynb) file in jupyter notebook and run all the cells </br>

</br></br>
## :key: Results 
#### Our model gave 98.51% accuracy for validation set of Pneumonia Detection via <code>tensorflow-gpu==2.0.0</code>
<br />
* The model has been trained on a python based environment on Jupyter platform.
* The model is iterated for a total epoch of 12. 
* The model has attained an accuracy of __98.51 %__ accuracy on the Validation set.

#### We got the following accuracy vs. epochs curve plot
![](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/blob/main/Outputs/accuracy%20vs%20epochs.png)<br />
#### The above figure shows the Accracy plot of the model throughout it's training journey. 

<br /><br />![](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/blob/main/Outputs/loss%20vs%20epochs.png)<br/>
#### The above figure shows the Loss plot of the model throughout it's training journey. 


## :clap: And it's done!
Feel free to mail me for any doubts/query 
:email: aryan.gupta18@vit.edu



## :handshake: Contribution
Feel free to **file a new issue** with a respective title and description on the the [Pneumonia_Detection](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/issues) repository. If you already found a solution to your problem, **I would love to review your pull request**!


## :heart: Owner
Made with :heart:&nbsp;  by [Aryan Gupta](https://github.com/beingaryan)


## :+1: Credits
* [https://www.pyimagesearch.com/](https://www.pyimagesearch.com/)
* [https://opencv.org/](https://opencv.org/)


## :handshake: Our Contributors
[CONTRIBUTORS.md](/CONTRIBUTORS.md)

## :eyes: Code of Conduct

You can find our Code of Conduct [here](/CODE_OF_CONDUCT.md).


## :eyes: License
MIT © [Aryan Gupta](https://github.com/beingaryan/Chest-Xray-Pneumonia-Prediction/blob/main/LICENSE)








