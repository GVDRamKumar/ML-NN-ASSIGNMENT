
# **CNN, RNN, and LSTM Tutorial for Machine Learning**

## **Overview**
This tutorial demonstrates how to implement and compare three fundamental types of neural networks: **Convolutional Neural Networks (CNN)**, **Recurrent Neural Networks (RNN)**, and **Long Short-Term Memory Networks (LSTM)** using **TensorFlow** and **Keras**. These models are applied to two well-known datasets:

1. **MNIST**: A dataset of handwritten digits (0-9) for image classification tasks.
2. **IMDB**: A dataset of movie reviews for sentiment analysis tasks.

The goal of this tutorial is to provide a clear understanding of how these different architectures work and how they can be applied to solve real-world machine learning problems.

---

## **Models Overview**

### **1. Convolutional Neural Networks (CNN)**
CNNs are used for image-related tasks. They use **convolutional layers** to extract spatial features from images. CNNs are ideal for tasks such as image classification, object detection, and more. The key operation in CNN is the **convolution operation**, which automatically detects patterns in the data.



### **2. Recurrent Neural Networks (RNN)**
RNNs are used for sequential data like time series, text, or speech. They maintain an internal state, which gets updated with each new input. RNNs are ideal for tasks where the order of data points is essential, such as sentiment analysis or language translation.



### **3. Long Short-Term Memory Networks (LSTM)**
LSTMs are a type of RNN designed to handle long-term dependencies and overcome the vanishing gradient problem. They are effective in processing sequences with long-term dependencies, such as text, speech, or time-series forecasting.


## **Datasets**

### **1. MNIST Dataset**
The **MNIST** dataset consists of 28x28 grayscale images of handwritten digits from 0 to 9. The goal is to classify each image into one of the 10 classes (0-9).

### **2. IMDB Dataset**
The **IMDB** dataset consists of movie reviews, labeled as either positive or negative. The task is to classify each review based on sentiment.

---

## **Data Preprocessing**

### **For CNN Model (MNIST)**
- **Reshaping**: The MNIST dataset consists of 28x28 grayscale images, which are reshaped to have dimensions (28, 28, 1) to match the input requirements of CNN layers.
- **Normalization**: Pixel values are scaled from the range [0, 255] to [0, 1] by dividing by 255.
- **One-hot Encoding**: Labels (digits) are one-hot encoded.

### **For RNN Model (IMDB)**
- **Padding Sequences**: The IMDB dataset consists of reviews with varying lengths. Padding ensures that all sequences have the same length.
- **Tokenization**: Words are converted into integer indices representing each word in the vocabulary.

### **For LSTM Model (MNIST)**
- **Reshaping for Sequence Processing**: The MNIST dataset is reshaped so each row of the image is treated as a timestep, with each pixel as a feature.
- **Normalization**: Pixel values are normalized, similar to the CNN model.

---

## **Model Building and Training**

- **CNN Model**: Convolutional layers followed by pooling layers to reduce the image dimensions. A dense layer is used for classification.
- **RNN Model**: Embedding and SimpleRNN layers for text data, followed by a sigmoid activation function for binary classification.
- **LSTM Model**: An LSTM layer followed by a dense layer for classifying digits.

---

## **Compiling the Models**

- **Optimizer**: Adam optimizer for adaptive learning rates.
- **Loss Function**: Categorical crossentropy for multi-class classification (CNN and LSTM), binary crossentropy for binary classification (RNN).
- **Metric**: Accuracy to evaluate performance.

---

## **Training the Models**

- **CNN and LSTM** are trained on the MNIST dataset.
- **RNN** is trained on the IMDB dataset.

---

## **Results and Visualization**

The models' accuracy and loss are plotted over training epochs, allowing for comparison of their performances.

---

## **Conclusion**

This tutorial demonstrates the use of CNN, RNN, and LSTM models for tasks like image classification (MNIST) and sentiment analysis (IMDB). Each model type is suited to different kinds of data, with CNNs excelling in image data, RNNs in sequential data, and LSTMs overcoming the limitations of traditional RNNs.

---

## **GitHub Repository**

- **Repository Link**: [https://github.com/GVDRamKumar/ML-NN-ASSIGNMENT.git](https://github.com/GVDRamKumar/ML-NN-ASSIGNMENT.git)

---

### **References**
1. TensorFlow and Keras Documentation
2. MNIST Dataset: LeCun, Y., et al., “Gradient-Based Learning Applied to Document Recognition,” Proceedings of the IEEE, 1998.
3. IMDB Dataset: Maas, A. L., et al., “Learning Word Vectors for Sentiment Analysis,” ACL, 2011.
