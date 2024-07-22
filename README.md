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
