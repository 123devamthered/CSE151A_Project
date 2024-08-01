# CSE151Arepoairlinegrp

## Data Preprocessing
To preprocess our data in our [Jupyter Notebook](https://github.com/123devamthered/CSE151Arepoairlinegrp/blob/main/milestone%202.ipynb), we examined the data frame and dropped the year column since all observations are from 2013, and the "name" columne since it is the same as the carrier column.  We also dropped the “hour”, “minute”, and “time_hour” columns to keep the consistency as they are only about the scheduled departure time but not the scheduled arrival time. We checked any columns having any NaN value and observed that some features have NaN values, so we dropped the rows that contain NaN values in the 'dep_time', 'arr_time', 'arr_delay', and 'air_time' columns. We examined the data types and found that some are floats and some are integers, so we converted them all to integers to be comparable.

## Column Descriptions
**id**: A unique identifier for each flight record in the dataset. <br/> 
**month**: The month in which the flight took place (1 to 12).  <br/> 
**day**: The day of the month on which the flight took place (1 to 31).  <br/> 
**dep_time**: The actual local departure time of the flight, in 24-hour format (hhmm).  <br/> 
**sched_dep_time**: The scheduled local departure time of the flight, in 24-hour format (hhmm). <br/> 
**dep_delay**: The difference between the actual and scheduled departure times of the flight, in minutes. A positive value indicates a delayed departure, while a negative value indicates an early departure.  <br/> 
**arr_time**: The actual local arrival time of the flight, in 24-hour format (hhmm).  <br/> 
**sched_arr_time**: The scheduled local arrival time of the flight, in 24-hour format (hhmm).  <br/> 
**arr_delay**: The difference between the actual and scheduled arrival times of the flight, in minutes. A positive value indicates a delayed arrival, while a negative value indicates an early arrival.  <br/> 
**carrier**: The two-letter code of the airline carrier for the flight.  <br/> 
**flight**: The flight number of the flight. <br/> 
**tailnum**: The unique identifier of the aircraft used for the flight.<br/> 
**origin**: The three-letter code of the airport of origin for the flight.<br/> 
**dest**: The three-letter code of the destination airport for the flight.<br/> 
**air_time**: The duration of the flight, in minutes.<br/> 
**distance**: The distance between the origin and destination airports, in miles.<br/> 
Source: https://www.kaggle.com/datasets/matinsajadi/flights

## Milestone 3
We have constructed an LSTM-based neural network RNN model to perform machine learning recursively.  It uses Long Short-Term Memory (LSTM) units to better capture and learn from sequential data. We have chosen LSTM networks because they are particularly effective for tasks involving time series, sequences, and other data with temporal dependencies, which our data are time sequential and might have temporal dependencies.<br/> 

For the train, validation, and test set, the ground truth is the actual arrival delay, and predictions are the arrival delays predicted by the model for the flights both in the corresponding train, validation, and test set. However, during the training, the model's performance is also validated using the test data. <br/> 

**Evaluate the model** <br/> 
After running our first model, the training loss/mse is about 2044.3040, with the testing loss/mse about 2014.2782.  For the trivial case, mse is calculated for the case that the mean of delay time is predicted for every flight. A trivial mse of approximately 1967.0509, is calculated, which is lower than both the training and testing mse, therefore implying that both the training error and testing error are high. The training error is higher than the testing error, which shows that there is no overfitting, but the model performance is unsatisfactory. The model is on the left side of the ideal place because both losses are high which is worse than trivial, so there might be an underfit. <br/> 

**Future models/improvements** <br/> 
We might experiment with changing the number of parameters and compare the performance of the model and also fine-tune the hyperparameters by increasing/decreasing the learning rate and see how it affects the mse. In terms of different models, if RNN continues to perform poorly, we might experiment with a multilayer perceptron neural network, returning to the basic neural network model to see if there will be any differences or improvements. We suspect that using RNN might be overcomplicating the calculation and problem. 

[Link to notebook for Milestone 3](https://colab.research.google.com/drive/1DPWlak3ZpMsrB1rbW8aA-p5RigMdIr5P?usp=sharing)

### Milestone 4 - Final Writeup
## INTRODUCTION
For our project, we will create a reinforcement learning model predicting arrival time delay based on different flight details like airline, scheduled departure and arrival time, departure time, distance, origin, etc. We feed in the data chronologically so our training cycle mimics the way our model would be deployed in the real world wherein the timestamp of new data would be later than old data. Therefore we choose RNN (recursive neural network) as our model. The model, if perfected, could be used on plane monitors to inform flyers about their real expected arrival time when already departed, based on the features mentioned above and some more. We would apply our model after the plane already takes off since one of the features is departure delay. The reason we chose this project was because we think that building a model that predicts arrival delays is very useful in today’s world. We think that it has a extensive application in the real world, where there are millions of flights taking place globally every year. Furthermore, our model is cool because it has important applications from both an industry or commercial perspective and from a philanthropic perspective. It can be used by different airlines to project the potential arrival delays of their flights and reduce/optimize on them. On the other hand, the model can be used to inform and update people about the late arrivals of their flights. A good predictive model, which would be effective at predicting flight arrival delays, could be used by both airlines and airports for the reasons mentioned above. 

## METHODS

# Data exploration / Preprocessing
We dropped the “name,” “year,” “hour,” “minute,” and “time_hour” columns but kept the “month” and “day” columns, because they are repetitive and insignificant to the model. We dropped null data instead of replacing it with the mean, median, or mode since we felt the dataset had enough observations remaining to train an accurate model. 
We then created the feature number of minutes departure is after 12AM on January 1st, 2013, because time progresses linearly, thus having a continuous time after a specific date summarizes all the information about the time of departure in a single value. This format makes the feature of time increase linearly instead of having jumps when the hour changes. We wrote a function to calculate the number of hours difference between the time zone of the origin airport and destination airport for each flight so that our model could take the actual time elapsed on a flight into consideration when predicting arrival delays.
 We one-hot encoded carrier, origin airport, and destination airport to make this data numerical instead of categorical.This ensures that each category is treated as distinct and independent without implying any ordinal relationship. This encoding helps improve model performance by allowing algorithms to process non-numerical features, such as text or labels, effectively and systematically. 
We also generated three pairplots of our data: one color-coding data points by carrier, one by origin airport, and one by destination airport, in order to visualize the relationships between multiple pairs of features in a dataset, allowing for a quick and comprehensive assessment of how variables interact with each other. 

# Model 1
We constructed a recurrent neural network with Tensorflow to perform machine learning recursively, as we are trying to create a reinforcement learning model. It uses Long Short-Term Memory (LSTM) layers to better capture and learn from sequential data. We have chosen LSTM networks because they are particularly effective for tasks involving time series, sequences, and other data with temporal dependencies, which our data are time sequential and might have temporal dependencies. We further researched how to create an LSTM RNN and found that Dropouts, and Batch Normalization are common components used with the keras Sequential model, which we included in our first model. We also chose ReLU to be our activation function for all layers as it has been found to significantly improve the performance and convergence speed of neural networks, and it is a popular choice in modern deep learning applications.

Our initial model uses the keras Sequential models.
```model = Sequential()```

The first hidden LSTM layer has 128 nodes and a ReLU activation function. It is followed by a Dropout layer with a dropout value of 0.2 as well as a Batch Normalization layer.
```
model.add(LSTM(128, input_shape=[1,129], activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
```

The second hidden LSTM layer also has 128 nodes and a ReLU activation function. It is followed by a Dropout layer with a dropout value of 0.1 as well as a Batch Normalization layer.
```
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
```

The third hidden layer is a Dense layer with 32 nodes and a ReLU activation function. It is followed by a Dropout layer with a dropout value of 0.2.
```
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
```

The Dense output layer has 10 nodes and a softmax activation function.
```model.add(Dense(10, activation='softmax'))```

We used an Adaptive Moment Estimation (Adam) optimizer with a learning rate of 0.001.
```opt = tf.keras.optimizers.Adam(learning_rate=0.001)```
For the train, validation, and test set, the ground truth is the actual arrival delay, and predictions are the arrival delays predicted by the model for the flights both in the corresponding train, validation, and test set. During the training, the model's performance is also validated using the test data.
We trained the model for 3 epochs then compared this to a trivial model that always predicts the mean arrival delay.
Link to Colab Notebook Containing Initial Model:
https://colab.research.google.com/drive/1DPWlak3ZpMsrB1rbW8aA-p5RigMdIr5P?usp=sharing 

# Model 2
In our second model, we changed the Dense output layer to have 1 node and use linear activation function as the output should be one continuous value(which is time). We found out a significant error in model 1 that mistakenly have our output layer to be 10 nodes with a softmax activation function that was intended for classification instead of regression. Which, it is the main cause of high error/loss in our mse for our Model 1.
Then we perform hyperparameter tuning to further improve our second model. We optimized the number of layers (tested 3, 4, and 5 layers), the activation function of the hidden layers (tested sigmoid, tanh, and ReLU), the number of nodes in each hidden layer (tested integers 15 to 150), whether each hidden layer was Bidirectional (since that is another degree of choice in building an RNN), whether to use LSTM or GRU layers, the dropout value (tested 0.0001 to 1 using logistic sampling), whether to use Batch Normalization after each hidden layer, the learning rate (tested 1e-4 to 1e-2 using logistic sampling), and the optimizer (tested SGD, Adam, and RMSProp). We ran Random Search on 500 trials.
Link to Colab Notebook Containing Second Model and Hyperparameter Tuning: https://colab.research.google.com/drive/1_FggJppGpTRlgajB7-8VfgpIBMBPLnC5?usp=sharing

## RESULTS
# Data Exploration / Preprocessing
Below, we include three pairplots and a correlation matrix generated during data preprocessing as well as an analysis of each figure.

