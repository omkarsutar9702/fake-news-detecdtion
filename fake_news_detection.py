#import data set
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
#read data set
df=pd.read_csv("D:/R files/alarm/alarm/news.csv")
df.shape
df.head()

#get labels 
labels=df.label
labels.head()

#split data into training and testing data
x_train , x_test , y_train , y_test = train_test_split(df['text'] , labels,
test_size=0.2 , random_state = 10)

#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


#build the confusion matrix
confusion_matrix(y_test,y_pred,labels=["FAKE" , "REAL"])

