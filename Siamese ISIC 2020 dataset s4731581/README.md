# Classifying Melanoma with Siamese Network and ResNet18 using Resized ISIC 2020 Challenge Dataset
![header](https://github.com/user-attachments/assets/41b71a67-a45e-4183-a437-9d65e9aa4461)

## Problem
The ISIC 2020 challenge dataset is a dataset of images of skin cancer (Melanoma). This project addresses binary melanoma recognition on a publicly available resized ISIC 2020 challenge dataset. The objective is to learn an image similarity function that separates malignant from benign lesions and to report accuracy and ROCAUC on a separated and balanced test set.  

## Dataset source
The dataset is from Kaggle: https://www.kaggle.com/datasets/ziadloo/isic-2020-256x256 (resized ISIC-2020). Use subject to the dataset’s terms and ISIC license. No personal data were present in the dataset, only images and labels are used. 

## Siamese network
Unlike traditional CNNs that classifies images using labels and predicting classes directly, the Siamese network learns a similarity function and uses that to perform classification. Two identical encoder networks with shared wrights were used to turn the images into L2 normalised embeddings, which is them compared cosine similarity. The training feeds three combinations of image pairs including positive-negative, negative-negative and positive-positive. The network then uses a binary loss to learn the difference between these pairs, thus telling the positive and negative images apart.  
<img width="629" height="265" alt="image" src="https://github.com/user-attachments/assets/8ca22b39-c1a0-48c5-b921-03186b8832c1" />

(Nag, 2022) 

https://www.kaggle.com/datasets/ziadloo/isic-2020-256x256
https://www.kaggle.com/datasets/ziadloo/isic-2020-256x256

## ResNet 18
ResNet18 is a residual CNN with an 18 layer setup and uses skipping connections to improve image training. It is made up of 3 by 3 convolution layers and max pools. The use of skipping connections helps the gradients to flow though the network easier, allowing for more efficient training. (Residual Networks (ResNet) - Deep Learning, 2025) 

## Implementation
The dataset is first imported via the Kagglehub library import commands available on the official ISIC 2020 dataset website. For simplicity a community resized version was used. Light augmentation such as random horizontal and vertical flips were done before training and normalised using the ImageNet channel (standard weights for ResNet encoders).  

Labels provided in the csv file from the dataset were assigned to the images via filename, and the data is split into stratified training, validating and testing sets, each containing positive-negative, negative-negative and positive-positive image pairs. Cosine similarity was used in the ResNet 18 encoder alongside with binary cross entropy, AdamW and mixed precision. Hard negative mining replaces some of the positive-negative pairs with deliberately confusing pairs to ensure the model is not learning entirely on easy pairs.  

After each epoch the training images was embedded with the average embeddings per class, which was then used to compare which embedding is closer using cosine similarity in validation accuracy and ROCAUC calculations. Each epoch was saved with the accuracy, loss and AUCROC score of training and validation set. Testing was done every other epoch to save computational resources.   

## Script structure
modules.py — model components including Siamese and ResNet18 encoder, cosine similarity and EMA modules 

dataset.py — dataset download from Kaggle and parsing using provided CSV, image/file mapping and stratified splits 

train.py — training, validation testing, saving, plotting, and early stop ability.  

predict.py — example usage of a trained checkpoint 

README.md — this document 

## Environment and setup
Python 3.13 

CUDA 12.8 supported Pytorch 

pandas numpy scikit-learn pillow tqdm kagglehub 

install dependencies via pip: 

```
pip install pandas numpy scikit-learn pillow tqdm kagglehub 

pip install pip3 install torch torchvision --index-url 
https://download.pytorch.org/whl/cu128 
```

## Results 
The testing accuracy got to 0.75 at around 20 epochs before dropping down to 0.65 later on. This is most likely due to the model overfitting the training data since validation accuracy stayed around 0.9 and kept increasing after 20 epochs whilst train losses also remained very low after 20 epochs.  
<img width="625" height="353" alt="image" src="https://github.com/user-attachments/assets/60dd4563-56ea-4fce-9d7e-9d1991e97748" />
<img width="643" height="739" alt="image" src="https://github.com/user-attachments/assets/2fe35961-eb18-443f-a09c-f17d64721ed7" />

Validation pair loss measures how the model distinguishes between same or different class when processing validation images. Early epochs with high values indicating an high initial learning rate may overshoot between epochs, flatten and suddenly peaking 

at around 20 epoch may suggest some overfitting occurring in the training data.  
<img width="620" height="360" alt="image" src="https://github.com/user-attachments/assets/cb183aec-7017-4e61-acd3-d378e1ff8444" />

AUCROC measures the probability of a false positive from the trained data. Missing data in the validation result is due to occasional NaN readings from the model, which is the result of missing class in the evaluated subset. The AUCROC result of both validation and testing data showed a decreasing trend at around 20 epoch, suggesting overfitting occurring.  
<img width="685" height="726" alt="image" src="https://github.com/user-attachments/assets/118b59f2-bc84-4e4c-a710-6da9a810de41" />

## Reproducibility
Use seed 67, current results trained on Nvidia RTX 2080 8GB, expected around 0.7 test accuracy in 2 to 3 hours of training, with default parameter settings currently inside train.py. 

## Improvements
Future improvements should aim to keep test performance stable and avoid overfitting too early in the training process. Use a simple learning rate schedule such as keeping ut stable for a few epochs then decrease gradually, enable EMA by default, and use slightly higher weight deca. Apply HNM only after a short warm up and at a modest fraction and use a bigger dataset.  

## References
Nag, R. (2022, Nov). A Comprehensive Guide to Siamese Neural Networks. Retrieved 

from Medium. 

Residual Networks (ResNet) - Deep Learning. (2025, Jul). Retrieved from geeksforgeeks: 
https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-
learning/ 
