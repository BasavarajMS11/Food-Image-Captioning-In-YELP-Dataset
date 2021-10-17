# Food-Image-Captioning-In-YELP-Dataset


## Problem Statement: 
To deep learning build a model that can describe the content of the food image in the form of text given an input image of food.

## Methodology :
The proposed methodology for food image captioning consists of three main modules.
<br/>
![alt text](https://github.com/BasavarajMS11/Food-Image-Captioning-In-YELP-Dataset/blob/master/Images/Methodology.JPG?raw=true)
<br/>
1. CNN-LSTM Caption Prediction(CLCP)
2. Multi Label Classification(MLC)
3. Natural Language Generator(NLG)

### Requirements:
- Python >3.7
- TensorFlow>2.0
- Google Transformer T5
- Spacy

### Dataset Required:
- Yelp Food Dataset
- Yummly28k
- FFoCat dataset



## Configuring .ipynb and .py files for inference:
 
## 1.CNN-LSTM Caption Prediction Module:
IPYNB File: CNN_LSTM_MultiLabel.ipynb
### 1.1 Training
It is the initial model where for the training the images with captions are used.
For this module a JSON file with image_id and caption are needed. Along with the images in a specific folder.
The json file is red and the captions are pre-processed. And a word embedding is created from converting word to vector.
The images from the path are considered for CNN feature extraction.
For the LSTM words from caption are provided sequentially.
LSTM tries to learn to predict next word from previous.
The loss is calculated at the end and the learning rate is predicted.

### 1.2 Testing
The image from the folder has to be read and passed to CNN feature extractor the same is passed to LSTM to generate caption by comparing the vector with the word2vec embedding.

## 2.Multi Label Classification Module
PY File: Mulit-label//food_category_classification.py
### 2.1 Training
It is a transfer learning approach.
InceptionV3 followed by dense layer is used to train the model.
The FFoCat dataset with 156 labels is used for training.

### 2.2 Testing 
Set the path of trained model in the model load section and the path that the image has to be read from.
Then run the code to get the desired output.

## 3.NLG
IPYNB File: NLG_Googles_T5_for_T2T.ipynb
### 3.1 Training;
Trained on C4 data and fine-tuned an yummly28 dataset.
The required caption must be present in a data frame and fed to the model for training.

### 3.2 Testing:
The output from Multilabel and CNN-LSTM must be concatenated and fed into the trained model to get the required output.

## Results

### Table 1.1: Results of Caption Generation on Yummly28k data
In this section we aim to generate the caption for the images from Yummly28k dataset. The caption that we try to generate must be close to the given caption in order to evaluate using some evaluation metrics discussed in above section.
The food images considered in Table 1.1 are from Yummly28k dataset. Given caption is the ground truth caption for the food image. Intermediate caption is output of CNN LSTM model. Intermediate Label is output of Multi Label Classifier. Generated caption is the final generated caption.
![alt text](https://github.com/BasavarajMS11/Food-Image-Captioning-In-YELP-Dataset/blob/master/Images/ResultsYummly28k_1.JPG?raw=true)
<br/>
![alt text](https://github.com/BasavarajMS11/Food-Image-Captioning-In-YELP-Dataset/blob/master/Images/ResultsYummly28k_2.JPG?raw=true)
<br/>

### Table 1.2: Performance evaluation metrics on Yummly28k data
By our proposed approach of caption generation different performance evaluation metrics achieved in Table 1.2 on Yummly28k dataset.<br/>
![alt text](https://github.com/BasavarajMS11/Food-Image-Captioning-In-YELP-Dataset/blob/master/Images/Perf_Evaluation_Yummly.JPG?raw=true)
<br/>

## Results on Yelp food dataset
In this section we attempt to generate caption for the YELP food data. There are two ways in which we generate the caption for YELP data. One for the uncaptioned data (Table 1.3) where the caption is generated for the image without caption. In other way (Table 1.4) the caption is generated for the image with irrelevant caption for which we try to provide better caption than the given caption.

### Table 1.3: Results of Caption Generation on Yelp uncaptioned food data
The images considered in Table 1.3 are from YELP uncaptioned food dataset. For the images considered there is no caption given in the dataset. The model is able to generate good captions for the images by effectively making use of intermediate caption and label prediction followed by NLG sentence generator for proper caption generation. The image and the caption generated have positive correlation between them.<br/>
![alt text](https://github.com/BasavarajMS11/Food-Image-Captioning-In-YELP-Dataset/blob/master/Images/ResultsUncapYelp.JPG?raw=true)
<br/> 

### Table 1.4: Results of Caption Generation on Yelp captioned food data
The images considered in Table 1.4 are from YELP captioned dataset. For the images considered there is caption given in the dataset. Here we aim to generate a better caption than the given that well describes the image. Since for some amount of data in the yelp for which caption is provided which is irrelevant. The model is able to generate good captions for the images by effectively making use of intermediate caption and label prediction followed by NLG sentence generator for proper caption generation. The image and the caption generated have positive correlation between them.<br/>
![alt text](https://github.com/BasavarajMS11/Food-Image-Captioning-In-YELP-Dataset/blob/master/Images/ResultsCapYelp.JPG?raw=true)
<br/>





 






