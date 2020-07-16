# for changing data format
from datetime import datetime
import dateutil.parser as dp

todate = dp.parse('2020-10-14')

datetimeobject = datetime.strptime(str(todate),'%Y-%m-%d %H:%M:%S')
newformat = datetimeobject.strftime('%m-%d-%Y')
print (newformat)

fhuad = lambda f: f.year

fhuad(newformat)

# to test pd.merge if it will give what i expect
d1 = {'id' : [1, 2, 3, 4, 5],
      'fruits' : ['orange', 'mango', 'apple', 'cashew', 'banana'],
      'veg' : ['carrot', 'cucumber', 'lettuce', 'yam', 'cabbage']}

d2 = {'id' : [1,1,1,1,2,2,2,3,3,3,4,4,4,4],
      'price' : [20,20,20,20,30,30,30,5,5,5,10,10,10,10]}

vitamins = pd.DataFrame(data=d1, columns=['id', 'fruits', 'veg'])
eaten = pd.DataFrame(data=d2, columns=['id', 'price'])

eat = pd.merge(vitamins, eaten, on='id', how='inner')
eat


# To find the best degree of polynomial

rmses = []
degrees = np.arange(1, 11)
min_rmse, min_deg = 5000, 0

for deg in degrees:

    # Train features
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly_train = poly_features.fit_transform(train_cop_transformed1)

    elastic_model1 = ElasticNet()
    kfold = KFold(n_splits=3, random_state=seed)
    en_cv_results = cross_val_score(elastic_model1, x_poly_train, train_coY, cv=kfold, scoring=scoring, n_jobs=-1)

    # Keeps the mean rmse of each polynomial iteration
    cv_rmse = np.sqrt(-en_cv_results)
    mean_cv_rmse = np.mean(cv_rmse)
    rmses.append(mean_cv_rmse)

    # Cross-validation of degree
    if min_rmse > mean_cv_rmse:
        min_rmse = mean_cv_rmse
        min_deg = deg

# Plot and present results
print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))

# plot the graph
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(degrees, rmses)
ax.set_yscale('log')
ax.set_xlabel('Degree')
ax.set_ylabel('RMSE')


# this sklearn transformer didn't work
# Build a custom Sklearn transformer to perform all preparation before modeling

# Build a custom sklearn transformer with 3 functions to prepare the data (drop features, replace 1, replace 2 and convert host to age.)

# import relevant libraries
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TidyData(BaseEstimator, TransformerMixin):
    def __init__(self, train_data):
        self.train_data = train_data
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        #function for creating age variable from DOB
        def create_age(data=X):
            X['age'] = X['DOB'].map(lambda x: x.year)
            X['age'] = 2015 - X['age']
            return X

        # function to correct misspelled gender names
        def correct_misspelled_gender():
            X['gender'].replace({'U' : 'Female', '247' : 'Female',
                                    'F' : 'Female', 'Femal' : 'Female', 'M' : 'Male'}, inplace=True)
            return X

        # function to correct misspelled state names
        def correct_misspelled_state():
            X['state'].replace({'Victoria' : 'VIC', 'New South Wales' : 'NSW'}, inplace=True)
            return X

        # convert all category variables' values to discrete numbers
        def discretize_categories():
            # gender variable
            X['gender'].replace({'Female' : 1, 'Male' : 2}, inplace=True)
            # job industry variable
            X['job_industry_category'].replace({'Entertainment' : 1, 'Telecommunications' : 2,
                                                   'IT' : 3, 'Manufacturing' : 4, 'Financial Services' : 5,
                                                   'Retail' : 6, 'Health' : 7, 'Property' : 8, 'Argiculture' : 9}, inplace=True)
            # state variable
            X['state'].replace({'NSW' : 1, 'VIC' : 2, 'QLD' : 3}, inplace=True)
            # wealth segment variable
            X['wealth_segment'].replace({'Affluent Customer' : 1, 'Mass Customer' : 2, 'High Net Worth' : 3}, inplace=True)
            # owns a car variable
            X['owns_car'].replace({'Yes' : 1, 'No' : 2}, inplace=True)
            return X

        # function to drop irrelevant variables
        def drop_irrelevant_variables():
            X.drop(['customer_id', 'first_name', 'last_name',
                       'DOB','job_title', 'deceased_indicator', 'address', 'postcode', 'country'],
                      axis=1, inplace=True)
            return X



        # return all variables as pandas DataFrame
        return X
