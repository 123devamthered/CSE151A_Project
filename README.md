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
