# Milestone 2: Data Preprocessing
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

# Milestone 3: Initial Model
We have constructed an LSTM-based neural network RNN model to perform machine learning recursively.  It uses Long Short-Term Memory (LSTM) units to better capture and learn from sequential data. We have chosen LSTM networks because they are particularly effective for tasks involving time series, sequences, and other data with temporal dependencies, which our data are time sequential and might have temporal dependencies.<br/> 

For the train, validation, and test set, the ground truth is the actual arrival delay, and predictions are the arrival delays predicted by the model for the flights both in the corresponding train, validation, and test set. However, during the training, the model's performance is also validated using the test data. <br/> 

## Evaluate the model
After running our first model, the training loss/mse is about 2044.3040, with the testing loss/mse about 2014.2782.  For the trivial case, mse is calculated for the case that the mean of delay time is predicted for every flight. A trivial mse of approximately 1967.0509, is calculated, which is lower than both the training and testing mse, therefore implying that both the training error and testing error are high. The training error is higher than the testing error, which shows that there is no overfitting, but the model performance is unsatisfactory. The model is on the left side of the ideal place because both losses are high which is worse than trivial, so there might be an underfit. <br/> 

## Future models/improvements
We might experiment with changing the number of parameters and compare the performance of the model and also fine-tune the hyperparameters by increasing/decreasing the learning rate and see how it affects the mse. In terms of different models, if RNN continues to perform poorly, we might experiment with a multilayer perceptron neural network, returning to the basic neural network model to see if there will be any differences or improvements. We suspect that using RNN might be overcomplicating the calculation and problem. 

[Link to notebook for Milestone 3](https://colab.research.google.com/drive/1DPWlak3ZpMsrB1rbW8aA-p5RigMdIr5P?usp=sharing)

# Milestone 4: Final Writeup
## Introduction
For our project, we will create a reinforcement learning model predicting arrival time delay based on different flight details like airline, scheduled departure and arrival time, departure time, distance, origin, etc. We feed in the data chronologically so our training cycle mimics the way our model would be deployed in the real world wherein the timestamp of new data would be later than old data. Therefore we choose RNN (recursive neural network) as our model. The model, if perfected, could be used on plane monitors to inform flyers about their real expected arrival time when already departed, based on the features mentioned above and some more. We would apply our model after the plane already takes off since one of the features is departure delay. The reason we chose this project was because we think that building a model that predicts arrival delays is very useful in today’s world. We think that it has a extensive application in the real world, where there are millions of flights taking place globally every year. Furthermore, our model is cool because it has important applications from both an industry or commercial perspective and from a philanthropic perspective. It can be used by different airlines to project the potential arrival delays of their flights and reduce/optimize on them. On the other hand, the model can be used to inform and update people about the late arrivals of their flights. A good predictive model, which would be effective at predicting flight arrival delays, could be used by both airlines and airports for the reasons mentioned above. 

## Methods
### Data exploration and Preprocessing

We dropped the “name,” “year,” “hour,” “minute,” and “time_hour” columns but kept the “month” and “day” columns, because they are repetitive and insignificant to the model. We dropped null data instead of replacing it with the mean, median, or mode since we felt the dataset had enough observations remaining to train an accurate model. 

We then created the feature number of minutes departure is after 12AM on January 1st, 2013, because time progresses linearly, thus having a continuous time after a specific date summarizes all the information about the time of departure in a single value. This format makes the feature of time increase linearly instead of having jumps when the hour changes. We wrote a function to calculate the number of hours difference between the time zone of the origin airport and destination airport for each flight so that our model could take the actual time elapsed on a flight into consideration when predicting arrival delays.

We one-hot encoded carrier, origin airport, and destination airport to make this data numerical instead of categorical.This ensures that each category is treated as distinct and independent without implying any ordinal relationship. This encoding helps improve model performance by allowing algorithms to process non-numerical features, such as text or labels, effectively and systematically. 
We also generated three pairplots of our data: one color-coding data points by carrier, one by origin airport, and one by destination airport, in order to visualize the relationships between multiple pairs of features in a dataset, allowing for a quick and comprehensive assessment of how variables interact with each other. 

### Model 1
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

### Model 2

In our second model, we changed the Dense output layer to have 1 node and use linear activation function as the output should be one continuous value(which is time). We found out a significant error in model 1 that mistakenly have our output layer to be 10 nodes with a softmax activation function that was intended for classification instead of regression. Which, it is the main cause of high error/loss in our mse for our Model 1.

Then we perform hyperparameter tuning to further improve our second model. We optimized the number of layers (tested 3, 4, and 5 layers), the activation function of the hidden layers (tested sigmoid, tanh, and ReLU), the number of nodes in each hidden layer (tested integers 15 to 150), whether each hidden layer was Bidirectional (since that is another degree of choice in building an RNN), whether to use LSTM or GRU layers, the dropout value (tested 0.0001 to 1 using logistic sampling), whether to use Batch Normalization after each hidden layer, the learning rate (tested 1e-4 to 1e-2 using logistic sampling), and the optimizer (tested SGD, Adam, and RMSProp). We ran Random Search on 500 trials.

Link to Colab Notebook Containing Second Model and Hyperparameter Tuning:
https://colab.research.google.com/drive/1_FggJppGpTRlgajB7-8VfgpIBMBPLnC5?usp=sharing

## RESULTS
### Data Exploration and Preprocessing
Below, we include three pairplots and a correlation matrix generated during data preprocessing as well as an analysis of each figure.

![pairplot1][pairplot1.png]
![pairplot2][pairplot2.png]
![pairplot3][pairplot3.png]

From the first pairplot, in which we have set the hue of the pairplot to be the carrier, we see no distinct pattern or correlation between departure delays and arrival delays across different carriers, indicating variability in delay patterns among carriers. In the second pairplot, in which the hue is set to the “origin” airport of the flight,  we see that some origin airports exhibit distinct delay patterns. For example, some origins may have more frequent longer delays. How an airport was designed or how it handles air traffic control could lead to more or less congestion and delays. Lastly, for the third pairplot, in which the hue is set to the destination airport, the pattern of arrival delays appears more varied across destinations, suggesting that certain destinations are more prone to delays.

![correlation matrix][correlation_matrix.png]

Looking at the correlation matrix, we can see that the variable with the strongest relationship with our target variable, “arr_delay”, is the variable “dep_delay” which measures departure delay. Intuitively, this makes sense considering that if the departure delay of a flight changes, there is a very strong likelihood that the arrival delay of the same flight also changes. Thus, it would make sense that they have the strongest correlation with a value of +0.91, indicating that if the departure delay goes up, so does the arrival delay and if the departure delay goes down, so does the arrival delay. The weakest correlation between the target variable arrival delay and another variable is the one between arrival delay and the variable “day” which indicates the day of the month during which the flight took place. 

### Model 1
After training our initial model as described, we have arrived with a training loss/mse of about 2022.7766, with the testing loss/mse about 2100.8508. To evaluate the performance of our model, we created a trivial case, where the mse is calculated for the case that the mean of delay time is predicted for every flight. A trivial mse of approximately 1967.0509, is calculated, which is lower than both the training and testing mse, therefore implying that both the training error and testing error are high. The training error is higher than the testing error, which shows that there is no overfitting, but the model performance is unsatisfactory, or in other words, horrible. The model is on the left side of the ideal place in a fitting graph because both losses are high which is worse than trivial, so there might be an underfit.
With this unsatisfactory results, we have come to 2 conclusions, either there was a major error in our model, or building a LSTM RNN for our dataset wasn’t a good fit. Therefore after our first model, we seek to debug and improve our model parameters, run hyperparameter tuning, or create a new neural network to train on our dataset and compare the MSE.

![loss and accuracy of initial model][loss_accuracy.png]

The above plots show the history of loss and accuracy over the three epochs when our first model was being trained. Both plots show a random distribution, showing that our model was not performing well and was likely under fitted.

![training and validation mse of intial model][training_val_mse.png]

The above graph shows the training and validation MSE over the three epochs when our initial model was being trained. It shows that the validation MSE was constant at approximately 2100 while the training mse was constant at about 2022.

### Model 2
After hyperparameter tuning, our model had an MSE of 408.4965515136719. This final model had 3 bidirectional Gated Recurrent Unit (GRU) layers, each with a dropout rate of 0.00016219921637706627 and a ReLU activation function, and 1 Dense layer with a linear activation function. There were 128 nodes in our first layer, 40 in our next 2 layers and 1 in our last layer. Our best model also had no Batch Normalization, a learning rate of about 0.002, and used the Adam optimizer.

## Discussion
### Data Exploration and Preprocessing
During data exploration, we found that our dataset only included information on flights in 2013 taking off from three airports (John F. Kennedy International Airport, LaGuardia Airport, Newark Liberty International Airport) in the New York Metropolitan Area, which we felt was too narrow of a scope to accurately predict arrival delays for flights across all years and airports; however, this dataset ultimately had more observations and was cleaner than other datasets we could find. Additionally, we thought weather conditions of a flight would be a great predictor of delays, but could not find a dataset containing the weather at the origin and destination airports at a specific date and time.

While preprocessing the pandas DataFrame that contained our flights dataset, we dropped the “name,” “year,” “hour,” “minute,” and “time_hour” columns because they were repetitive but kept the “month” and “day” columns. We then created the feature number of minutes departure is after 12AM on January 1st, 2013 since it summarized all the information about the time of departure in a single value. Also, this format makes the feature of time increase linearly instead of having jumps when the hour changes. For example, the minute changing to noon in hhmm format would be 1159 and 1200, which has a 41 unit difference even though it is actually only a 1 minute difference. We dropped null data instead of replacing it with the mean, median, or mode since we felt the dataset had enough observations remaining to train an accurate model.

Originally, we kept the “tailnum” column since the tail number of airplanes indicate the specific model of airplane, and we hypothesized that some types of aircrafts could be more prone to experiencing delays.

We also generated three pairplots of our data: one color-coding data points by carrier, one by origin airport, and one by destination airport. They revealed a positive correlation between departure delay and arrival delay, somewhat predictably. Otherwise, the carrier, origin, and destination categorical variables were not separable in any of the scatterplots, and many of the scatterplots were rectangular shaped since features like month, day, schedule arrival time, scheduled departure time, and actual departure time are uniformly distributed.

Furthermore, we wrote a function to calculate the number of hours difference between the time zone of the origin airport and destination airport for each flight so that our model could take the actual time elapsed on a flight into consideration when predicting arrival delays. We also one-hot encoded carrier, origin airport, and destination airport to make this data numerical instead of categorical.

### Model 1

We tried to build a recurrent neural network, because we were aiming to use reinforcement learning. Recurrent neural networks are used for analyzing sequential data. For example, if we wanted to predict whether a company’s stock is going to increase or decrease, we could use a recurrent neural network because it considers the behavior of the stock in say, the last few days, to project the behavior of the stock tomorrow.

In this case, the narrow scope of our dataset actually proved useful in designing our model. Since there are only three origin airports in the dataset, the departure and arrival delays of flights happening at the same airports on the same day could potentially affect each other. To illustrate, if our model is trying to predict the arrival delay of a flight taking off from LaGuardia Airport on a certain day and there were a number of flights taking off from LaGuardia right before that were majorly delayed, we hypothesized that it would be prudent to predict a longer arrival delay.

We researched how to implement RNNs, and found various pieces of sample code that we consulted and weaved together and edited until we got no errors. We realized that the LSTM layer in tensorflow keras was an implementation of an RNN, and thus primarily used that.

We found that LSTM layers have a more complex architecture than GRU layers that allows it to take in more parameters and potentially learn more nuanced relationships in the data. Even though LSTM layers are slower to train than GRU layers, we chose to build our initial model with LSTM layers to give our model the capability to see long-term dependencies between features.

We used a combination of 128 and 32 nodes in our hidden layers because we thought having a large number of neurons in each layer would give our model the complexity needed to handle the several features of our dataset. We used the ReLU activation function in our hidden layers so that our model could perceive non-linear relationships. We used the Adam optimizer because it is an iterative gradient descent-based optimizer that can be applied to train all types of models, and we set the learning rate to 0.001 so that the step size would be small enough as to not overshoot the minimum cost.

We also included batch normalization because it is a technique often used to make learning faster, improve model performance, avoid overfitting, and reduce a model’s sensitivity to an arbitrarily set learning rate. Additionally, we included Dropout layers since they are a regularization method used with recurrent neural networks to prevent overfitting. It is generally stated that a dropout rate of 0.2 or 0.1 is suited for most tasks, so those are the values we used for our initial model.

Since we were implementing an RNN for the first time, it was inevitable that we would have an error in our code. In the Dense output layer of the model, we set the number of nodes to 10 and the activation function to softmax. Softmax is used for multi-class classification problems since it’s the aggregation of multiple sigmoid functions. We should have been using a linear activation function since we are doing regression and not classification. Also, we were only predicting one value, the arrival delay of a given flight, so the output layer should have only had one node. As a result of this mistake, our results on the testing set were worse than if we were to simply predict the training mean for everything.

Another possible reason why our initial model was severely underperforming could be that batch normalization layers without any modifications are actually not well-suited for recurrent neural networks. Although batch normalization improves performance for convolutional neural networks, we later read scientific studies showing that batch normalization must be modified in order to be used with recurrent neural networks. As a result, the use of batch normalization in our initial model could have worsened its performance; however, this issue is later addressed during hyperparameter tuning for our second model.

### Model 2

For our second model, we realized that it did not make sense for the output layer to have 10 nodes when we were predicting a single value, so we changed the last layer to have 1 node. Also, since softmax is best suited for classification tasks instead of regression, we changed the output layer’s activation function to be linear. 

To further improve our model, we did hyperparameter tuning with Random Search. Although Grid Search would be more thorough, we wanted to optimize as many hyperparameters as possible, which would take an extremely long time to do with Grid Search. We determined 500 trials of Random Search would be a good representation of the lowest MSE our model could achieve even if its methods are not as precise as Grid Search.

We decided to tune as many hyperparameters as we could. This resulted in testing all the combinations of hyperparameters detailed in the Methods section. We thought in addition to ReLU we could test sigmoid and tanh activation functions since another function could possibly be better suited to the relationships in our data. We also tried different levels of model complexity by trying to increase or decrease the number of hidden layers to 3, 4, or 5 or increase or decrease the number of neurons in each layer from 15 to 150. Bidirectional recurrent neural networks also utilize both the forward and backward processing of data, which means that they use data that is sequentially ordered both before and after the current prediction being made. We thought bidirectional layers could improve our model since knowing the delays of flight both before and after the flight we are trying to predict the arrival delay of would improve our prediction, as it would indicate whether delays at a specific airport are on an upward or downward trend. We also tried using LSTM and GRU layers because although they are very similar, GRU layers are often preferred for their simple design and efficiency in learning. The dropout rate, learning rate, type of optimizer, and batch normalization were other degrees of choice in designing our RNN that we thought it would be prudent to tune.

Our final hyperparameter-tuned model had an MSE of about 408, which is much lower than the trivial predictor’s MSE of about 1967, which we deemed a significant improvement. Our baseline for error was predicting the training mean for every instance of our testing set, which gave us an MSE of 1967, nearly quintupling our final MSE, so we sufficiently outperformed it. We also likely did not overfit, as our training and testing MSE are very close.

## Conclusion

Air travel has become increasingly significant in an increasingly globalized world, aided by technological advancements and reduced costs which makes it associable to the masses. However, a persistent challenge is the unreliability of consistent travel times. Our objective was to address this issue by developing a model designed to predict potential delays. Utilizing historical flight data, we created a machine learning model capable of forecasting delays. This predictive model not only aids airlines in optimizing their schedules but additionally it also improves the passenger’s experience by providing more accurate travel information. In essence, our plan was to diminish the uncertainty associated with air travel, thereby making it more dependable and less infuriating. 

In the future, we may explore alternative methods to further quantify its accuracy. One possible way is to assess accuracy by using thresholds. Rather than relying solely on Mean Squared Error (MSE), which can be challenging to interpret, we propose employing accuracy based on thresholds. Our current MSE is 408, but without a clear benchmark, it’s hard to determine if this is truly good. Instead, by setting thresholds, we can measure how well our model's predictions align with actual delays within specific time frames. This could be implemented as follows.

For each observation in our testing dataset, we compare the predicted delay to the actual delay and determine if it falls within a certain threshold. For example, using a threshold of 15 minutes, if the model predicts a flight will land at 4:02 pm and the actual landing time is 3:50 pm, the difference is 12 minutes, which is within our threshold. This would be counted as accurate (a score of 1). If the predicted time is more than 15 minutes off, it would be counted as inaccurate (a score of 0). We repeat this process for all observations, then calculate an accuracy score based on these 1s and 0s.

By applying different thresholds, such as 10, 15, and 30 minutes, we can provide a clear, quantifiable measure of our model’s performance. For example, we might report that our model has: 80% accuracy within 15 minutes, 90% accuracy within 25 minutes, and 98% accuracy within 35 minutes.
This approach may provide a more tangible and understandable evaluation of our model's efficacy, making it easier to communicate its reliability and effectiveness in predicting flight delays.

## Statement of Collaboration

Sydney:
* collaborated on hyperparameter tuning code with Lulu
* wrote function outputting hour offset between origin and destination airports 
* wrote function outputting whether a flight took off and landed on the same day
* wrote milestone 1 abstract


Lulu: Team Leader
* Write up for milestone 2 and 3
* collaborated on hyperparameter tuning code with Sydney
* ran hyperparameter tuning on SDSC


Genie:
* created figures for data exploration
* Wrote code for data pre-processing
* created feature vectors
* conducted research for initial model
* wrote code for initial model
* evaluated initial model 


Devam:
* Data preprocessing: wrote function converting hhmm format to minutes since Janruary 1st 2013
* Conducted data exploration to discover an inconsistency between our dataset columns
* Looked for supplemental datasets
* Troubleshooted hyperparameter tuning code


Aahil:
* found flights dataset
* wrote functions to one-hot encode carrier, destination and origin columns of unprocessed dataset
* Set up hyperparameter tuning on SDSC
* Looked for supplemental datasets



