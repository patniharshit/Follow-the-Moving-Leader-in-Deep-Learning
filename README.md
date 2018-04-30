# Follow-the-Moving-Leader-in-Deep-Learning

FTML is closely related to RMSprop and Adam. In particular, it enjoys their nice properties, but avoids their pitfalls. Experimental results on a number of deep learning models and tasks demonstrate that FTML converges quickly, and is always the best (or among the best) of the various optimizers.

We have experimented with a CNN model on MNIST dataset and LSTM model on tweets and facebook posts.

## Dependencies
* Tensorflow
* Keras
* Matplotlib
* Keras Contrib: https://github.com/keras-team/keras-contrib

## Data
https://drive.google.com/drive/folders/1P7qR5Nq4ll3AXqueQ-ta_o3bJMS3jnDj?usp=sharing

## To run cnn.py:
Download data from above drive link and data folder under the name input.
python3 code/cnn.py

## To run lstm.py:
Download lstm folder which contains all the data and code for lstm model.

python lstm/code/lstm.py [number of epochs]

## Reference:
* Follow the Moving Leader in Deep Learning, Shuai Zheng, James T. Kwok
http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf

## Team Members:
* Yash Verma        (201501103)
* Harshit  Patni    (201501107)
* Aayush Surana     (201531012)
* Lakshya  Agrawal  (201530102)
