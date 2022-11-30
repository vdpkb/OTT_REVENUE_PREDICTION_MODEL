# OTT_REVENUE_PREDICTION_MODEL

DESCRIPTION: 

Over The Top Platform is a direct to consumer video(media) content platform. There are several OTT platforms available around us including Netflix, Hulu, youtube, Disney Hotstar, Zee5, Amazon Prime Video, Jio Cinemas etc.,

There are different OTT revenue models : SVOD (Subscriber Video On Demand), AVOD(Advertising Video On Demand) and Hybrid
Here, an attempt is made to predict Netflix revenue based on number of subscribers.

The project predicts revenue of Netflix OTT platform using different models : Linear Regression, Decision Tree, Random Forest and KNN models.
At the end of the python notebook, comparision against models is made based on Mean Squared Error (MSE) parameter.

DATA DESCRIPTION:

Source: Download Netflix related data from here - https://www.kaggle.com/datasets/azminetoushikwasi/ott-video-streaming-platforms-revenue-and-users 

ContentSpend.csv
NumSubscribers.csv
NumSubscribersByRegion.csv
Profit.csv
Revenue.csv
RevenueByRegion.csv

Of these available csv files, the project uses ContentSpend, NumScubscribers, Profit and Revenue files are used
Independant variable X: Subscribers/Year/Content Spend/Profit  
Dependant variable Y: Overall revenue generated in dollars

PROCESS DESCRIPTION

-> Import required libraries from matplotlib, numpy, pandas, seaborn and sklearn
-> Import dataset from source link mentioned above
-> Visualize the dataset to understand the correlation, dependencies and trend (if any) between features
-> Deal with missing values (NaN)
-> Linear Regression Model
-> Random Forest Regression Model
-> KNN Regression Model
-> Decision Tree Regression Model
-> Compare the models

