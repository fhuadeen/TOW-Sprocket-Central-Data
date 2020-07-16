## Putting all together into a Pipeline for training our model and predicting

#### Create a custom sklearn transformer for data preparation
""" 1. remove unwanted variables
2. convert all U to mode female
3. convert dob to age
4. discretise categorical variables."""


# Build a custom Sklearn transformer to perform all preparation before modeling

# Build a custom sklearn transformer with 3 functions to prepare the data (drop features, replace 1, replace 2 and convert host to age.)

# import relevant libraries
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# assign column index
id_in, first_name_in, last_name_in, gender_in, past_3_years_bike_related_purchases_in, DOB_in, job_title_in, job_industry_category_in, wealth_segment_in, deceased_indicator_in, owns_car_in, tenure_in, address_in, postcode_in, state_in, country_in, property_valuation_in = range(0, 17, 1)

class TidyData(BaseEstimator, TransformerMixin):
    def __init__(self, data_columns):
        self.data_columns = data_columns
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        # to convert year format in Host Since variable to number of years
        X.iloc[:, DOB_in] = X.iloc[:, DOB_in].map(lambda x: x.year) # create the variable
        # dividing it by the year 2018 as we assume to be analysing for the year 2018
        X.iloc[:, DOB_in] = 2015 - X.iloc[:, DOB_in] # convert year to number of years (age)

        # convert U to female
        X.iloc[:, gender_in].replace({'U' : 'Female', '247' : 'Female', 'F' : 'Female', 'Femal' : 'Female', 'M' : 'Male'}, inplace=True)

        # convert 'Victoria' and 'New South Wales' to VIC and NSW respectively
        X.iloc[:, state_in].replace({'Victoria' : 'VIC', 'New South Wales' : 'NSW'}, inplace=True)

        # Label encode all categorical variables to prepare for estimating missing values
        X.iloc[:, gender_in].replace({'Female' : 1, 'Male' : 2}, inplace=True)
        X.iloc[:, job_industry_category_in].replace({'Entertainment' : 1, 'Telecommunications' : 2, 'IT' : 3, 'Manufacturing' : 4, 'Financial Services' : 5, 'Retail' : 6, 'Health' : 7, 'Property' : 8, 'Argiculture' : 9}, inplace=True)
        X.iloc[:, state_in].replace({'NSW' : 1, 'VIC' : 2, 'QLD' : 3}, inplace=True)
        X.iloc[:, wealth_segment_in].replace({'Affluent Customer' : 1, 'Mass Customer' : 2, 'High Net Worth' : 3}, inplace=True)
        X.iloc[:, owns_car_in].replace({'Yes' : 1, 'No' : 2}, inplace=True)

        # Drop irrelevant features
        X.drop(['customer_id', 'first_name', 'last_name', 'job_title', 'deceased_indicator', 'address', 'postcode', 'country', ], axis=1, inplace=True)

        # return all variables as pandas DataFrame
        return X


# function for running full transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def pipeline_transformer(num_features, cat_features):
    """ - To transform numerical variables and categorical variables separately
    and then concatenate them.
    - You will first create 2 separate lists of variable names for the numerical
    variables and categorical variables.
    - num_features = list of numerical variable names
    - cat_features = list of categorical variable names

    - depends on sklearn functions - Pipeline, StandardScaler, OneHotEncoder,
    ColumnTransformer, SimpleImputer which must be imported first
    """
    num_pipe = Pipeline([('Imputer', SimpleImputer(strategy='mean')), ('Scaler', StandardScaler())])
    cat_pipe = Pipeline([('Imputer', SimpleImputer(strategy='most_frequent')), ('Encoder', OneHotEncoder())])

    full_pipe = ColumnTransformer([('nums', num_pipe, num_features), ('cats', cat_pipe, cat_features)])

    return full_pipe


# If the class doesn't working again, try to change the attribute of the TidyData class to the columns of the data to be used.
"""1. i.e. def __init__(self, data_columns)
2. so when we call the class, it will be TidyData(data_columns=data_name.columns)"""
