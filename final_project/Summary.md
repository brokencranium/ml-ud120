The dataset contains the following features:  
['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'bonus', 'exercised_stock_options', 
'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 
'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 
'long_term_incentive', 'from_poi_to_this_person']  

Not all features are required to identify the POI, the type for POI is boolean. There are 143 datapoints,
and with such number its difficult to train the models. Also, there is quite some missing data. For this
project, 11 features were used, many of the features with sparse data were dropped out. For few records,
from_messages and to_messages do not stack with the from_poi_to_this_person and from_this_person_to_poi. 
This was probably some data error and had to be fixed by making few assumptions. Not the ideal way, but 
with no. of data points available, dropping instances was not an option other than the outliers.  

The dictionary key name of total and persons greater than 10000000$ of salary are considered as outlier
and they were removed from the training/testing dataset. 

Two new features were created, 'from_poi_ratio' and 'to_poi_ratio'. Since using just the number of messages
from poi and to poi does not give a good insight. My assumption was, if majority of the communication is done
with the poi's then the probability of the person being poi is high. Whenever there was lack of data 
for 'from_this_person_to_poi' and 'from_poi_to_this_person', and if the person was a POI, the POI ration
was set to 1.  Without these custom features, the precision value was 0.22284 and recall was at 0.20100.
On adding these features, precision and recall value improved to 0.5329 and 0.50600.  

Initially used PCA to reduce the number of features and then ran SVC, could not get both precision and 
recall to above 0.5. I used grid search to find the right number of components. The end result wasn't
satisfactory.Listed below are the combinations(hyperparameters) that I tried with PCA and SVC.
 
param_grid = dict(clf__kernel=['sigmoid', 'rbf'],  
                  clf__C=[0.001, 0.1, 1, 10, 100, 1000, 10000, 1e3, 5e3, 1e4, 5e4, 1e5],  
                  clf__gamma=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],  
                  reduce_dim__n_components=[1, 2, 4, 6, 8, 10, 12, 13])    

Using minmax scaler did not improve the result by much.  

 
I started this project using PCA with SVC, and tuning it with various hyper-parameters with Grid Search  
did not give the intended result. I started experimenting with Random Forest, this gave a precision value 
around 0.6 but the recall was still at 0.27. At this point I thought, maybe Random Forest needs lot more 
data to  for the model to function appropriately. I used a Decision Tree classifier even though I knew  
the result is going to be biased based on the nature of the algorithm. My goal was to get more than 0.3 
for precision and recall, so a slightly biased model was acceptable for me.  

Using a Decision Classifier gives a precision of 0.53 and a recall of 0.50. For this scenario, where the consequences 
are high, the precision value needs to be as high as possible. The model should not be identifying someone 
as a POI who actually isn't. The false positive should be as low as possible. Having a relatively higher 
false negative is still acceptable at the cost of not falsely accusing a person as POI. So a lower recall 
is acceptable for this problem.   

Random forest with cross validation enabled gives a precision of 0.6913 for the tests provided. Random 
forest by default uses 2/3rd of the data for training and 1/3rd for testing. Using CV enables all the 
data provided is used for training and testing. Since there isn't much data to generalize, the preferred 
max depth for the decision trees is set to none. Using grid search with cv=10 for the all the params listed in the 
param_grid did not avail good results with SVC with PCA. 

Initially, 70% of the data was held for training and 30% was done for testing. I suspected that there 
was'nt enough data points for a model to converge. I started using 90% for training and 10% for testing, 
this meant my test data was not a good indicator for the performance of the model. I did rely on the 
tester.py script to check my model performance.  

I have attached the spreadsheet containing the performance results for few of the models. Decision tree 
classifier resulted with precision and recall values of above 0.5. I personally prefer the Random Forest 
because its less biased and also the precision is above 0.6.  For this type of problems, a model with higher 
precision should be given more weight. 


poi_id.py - Contains Random Forest and Decision Tree Classifier.
poi_id.py - Contains PCA with SVC 
