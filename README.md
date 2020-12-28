# Predictive-analysis

Objective: The objective of the assignment is to make a predictive analysis, a multilabel classification of genres on the movie genre prediction dataset using Apache Spark.

Dataset: The training set consists of 31103 entries each with movie_id, movie_name, plot and genre. The movies can be classified into 20 genres. Each movie can be associated with multiple genres. The test set consists of 7776 entries each with movie_id, movie_name and plot.

Genre: Drama, Comedy, Romance Film, Thriller, Action, World Cinema, Crime Fiction, Horror, Black-and-white, Indie, Action/Adventure, Adventure, Family Film, Short Film, Romantic drama, Animation, Musical, Science Fiction, Mystery and Romantic comedy. 

Dependency:The execution of all the parts are dependent on Pandas. This library is used for reading the data from csv file without any overflow of data from one column to another as in the case of pyspark. Also, it was used to concatenate the final predictions from all 20 classifiers.

Note: The paths of the data files need to be adjusted for reading.


Part – 1
Objective: The goal here is to analyze the data and use the information provided in the plot of the movies by extracting features and training a machine learning model to predict genres of the movies. Using this machine learning model, predict the genres associated with the movies in test dataset.

Approach: Here we train 20 binary classifiers, one classifier for each genre, concatenate the predictions from all the binary classifiers which provides the1.  illusion of a multilabel classifier.

Methodology: 
  First step is to read the train.csv into a dataframe. But since reading the csv using spark directly caused over flow of data from plot column to genre column, we came up with the approach of reading the data into a pandas dataframe [9] and convert it into a spark dataframe using SQLContext() [1] to avoid the corruption of data as explained earlier. 
  In the dataset, the important information lies in the plot of the movie which determine the genre of the movie; hence the features are extracted from the column ‘plot’. In order to separate irrelevant data, the plots are type casted to lower case and any character other than 26 letters of English alphabet a-z are removed. This results in the plots containing only clean words. 
  The plots are in terms of brief summary of the movie. In order to extract the features, the words are tokenized using Tokenizer() [2]. To avoid insignificant tokens that do not add much to the training process, the stop words likes an, the, for etc. are removed using StopWordsRemover() [3] from the tokens. Since the number of stems obtained is huge, the most frequent 9000 words that occur in at least 5 movie plots are determined and vectorized count is obtained using CountVectorizer() [5]. These act as the feature vectors for training the model.
  A user defined function get_predictions() is defined which takes in parameters like dataset and a genre. A column of labels is created based to the genre parameter which holds a value of 0 if none of the genres of a particular movie is the genre the classifier is based on and 1 otherwise. The function also defines a Logistic regression model and trains the model using the dataset provided as the input using the labels generated as explained above. 
  The model is then used to make predictions on the test data for all 20 genres. All the corresponding 20 predictions are concatenated to mimic the result of a multilabel classifier prediction. The concatenation process was quiet tedious using spark, hence we used pandas to achieve the joining of predictions into a single column using concat_ws() [8] and then converted back to a spark dataframe. All the logical transformations are applied using pyspark functions. The final combined prediction is written into a csv file.

Score: On submitting this prediction on Kaggle, we received a public score of 0.94002.


Part – 2
Objective: The goal is to improve the performance from the previous model by implementing Term Frequency-Inverse Document Frequency (TF-IDF), a feature engineering technique that enhances the significance of plot in training the model.

Approach: Here also we train 20 binary classifiers, one classifier for each genre, concatenate the predictions from all the binary classifiers which provides the illusion of a multilabel classifier. In addition, before the feature vectors are used to train the classifiers, the features are rescaled using the inverse document frequency. 

Methodology: 
  First step is to read the train.csv into a dataframe. But since reading the csv using spark directly caused over flow of data from plot column to genre column, we came up with the approach of reading the data into a pandas dataframe and convert it into a spark dataframe using SQLContext() [1] to avoid the corruption of data as explained earlier. 
  In the dataset, the important information lies in the plot of the movie which determine the genre of the movie; hence the features are extracted from the column ‘plot’. In order to separate irrelevant data, the plots are type casted to lower case and any character other than 26 letters of English alphabet a-z are removed. This results in the plots containing only clean words. 
The plots are in terms of brief summary of the movie. In order to extract the features, the words are tokenized using Tokenizer() [2]. To avoid insignificant tokens that do not add much to the training process, the stop words likes an, the, for etc. are removed using StopWordsRemover() [3] from the tokens. Since the number of stems obtained is huge, the most frequent 9000 words that occur in at least 5 movie plots are determined and vectorized count is obtained using CountVectorizer() [5]. These act as the feature vectors for training the model.
  The feature vector can be better engineered by using inverse document frequency, which in turn removes the least significant features increasing the weightage of the more important features based on their frequency in the minimum number of documents they appear in. This functionality is implemented using IDF() [6], whose standard formula is idf = log((m + 1) / (d(t) + 1)), where m is the total number of documents and d(t) is the number of documents that contain term t [7].
  Then the user defined function get_predictions() is defined which takes in parameters like dataset and a genre. A column of labels is created based to the genre parameter which holds a value of 0 if none of the genres of a particular movie is the genre the classifier is based on and 1 otherwise. The function also defines a Logistic regression model and trains the model using the dataset provided as the input using the labels generated as explained above. 
  The model is then used to make predictions on the test data for all 20 genres. All the corresponding 20 predictions are concatenated to mimic the result of a multilabel classifier prediction. The concatenation process was quiet tedious using spark, hence we used pandas to achieve the joining of predictions into a single column using concat_ws() [8] and then converted back to a spark dataframe. All the logical transformations are applied using pyspark functions. The final combined prediction is written into a csv file.

Score: On submitting this prediction on Kaggle, we received a public score of 0.97973.


Part – 3
Objective: The goal is to further improve the performance of the model implemented in part – 2 by using any of the modern text-based feature engineering methodologies such as Word2vec, Glove, Doc2vec, Topic Modelling etc. 

Approach: Here too we train 20 binary classifiers, one classifier for each genre, concatenate the predictions from all the binary classifiers which provides the illusion of a multilabel classifier. In addition, before the feature vectors are used to train the classifiers, they are passed to HashingTF to create a better feature vector to train the classifier on.

Methodology: 
  First step is to read the train.csv into a dataframe. But since reading the csv using spark directly caused over flow of data from plot column to genre column, we came up with the approach of reading the data into a pandas dataframe and convert it into a spark dataframe using SQLContext() [1] to avoid the corruption of data as explained earlier. 
  In the dataset, the important information lies in the plot of the movie which determine the genre of the movie; hence the features are extracted from the column ‘plot’. In order to separate irrelevant data, the plots are type casted to lower case and any character other than 26 letters of English alphabet a-z are removed. This results in the plots containing only clean words. 
The plots are in terms of brief summary of the movie. In order to extract the features, the words are tokenized using Tokenizer() [2]. To avoid insignificant tokens that do not add much to the training process, the stop words likes an, the, for etc. are removed using StopWordsRemover() [3] from the tokens. Then use the HashingTF() [4], to create the term frequency vector. These vectors act as the features for training the model.
  Then the user defined function get_predictions() is defined which takes in parameters like dataset and a genre. A column of labels is created based to the genre parameter which holds a value of 0 if none of the genres of a particular movie is the genre the classifier is based on and 1 otherwise. The function also defines a Logistic regression model and trains the model using the dataset provided as the input using the labels generated as explained above. 
  The model is then used to make predictions on the test data for all 20 genres. All the corresponding 20 predictions are concatenated to mimic the result of a multilabel classifier prediction. The concatenation process was quiet tedious using spark, hence we used pandas to achieve the joining of predictions into a single column using concat_ws() [8] and then converted back to a spark dataframe. All the logical transformations are applied using pyspark functions. The final combined prediction is written into a csv file.

Score: On submitting this prediction on Kaggle, we received a public score of 0.98906.

Reference:
[1] https://spark.apache.org/docs/1.6.1/sql-programming-guide.html
[2] https://spark.apache.org/docs/latest/ml-features#tokenizer
[3] https://spark.apache.org/docs/latest/ml-features#stopwordsremover
[4] http://en.wikipedia.org/wiki/Feature_hashing
[5] https://spark.apache.org/docs/latest/ml-features#countvectorizer
[6] https://spark.apache.org/docs/latest/ml-features#tf-idf
[7] https://spark.apache.org/docs/2.2.0/api/java/org/apache/spark/mllib/feature/IDF.html
[8] https://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html
[9] https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html
