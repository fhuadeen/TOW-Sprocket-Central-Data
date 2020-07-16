# create age variable from DOB
def create_age(data=None):
    """ To create age variable from DOB. """
    data['age'] = data['DOB'].map(lambda x: x.year)

    data['age'] = 2015 - data['age']

    return data


# correct misspelled gender
def correct_misspelled_gender(data=None):
    """ To correct misspelled gender values"""
    data['gender'].replace({'U' : 'Female', '247' : 'Female',
                              'F' : 'Female', 'Femal' : 'Female', 'M' : 'Male'}, inplace=True)
    return data

# correct misspelled state
def correct_misspelled_state(data=None):
    """ To correct misspelled state values"""
    data['state'].replace({'Victoria' : 'VIC', 'New South Wales' : 'NSW'}, inplace=True)

    return data


# converts categorical variables to discrete numbers
def discretize_categories(data=None):
    """ Converts all categorical variables' values to discrete numbers"""
    # gender variable
    data['gender'].replace({'Female' : 1, 'Male' : 2}, inplace=True)
    # job industry variable
    data['job_industry_category'].replace({'Entertainment' : 1, 'Telecommunications' : 2,
                                           'IT' : 3, 'Manufacturing' : 4, 'Financial Services' : 5,
                                           'Retail' : 6, 'Health' : 7, 'Property' : 8, 'Argiculture' : 9}, inplace=True)

    # state variable
    data['state'].replace({'NSW' : 1, 'VIC' : 2, 'QLD' : 3}, inplace=True)

    # wealth segment variable
    data['wealth_segment'].replace({'Affluent Customer' : 1, 'Mass Customer' : 2, 'High Net Worth' : 3}, inplace=True)

    # owns a car variable
    data['owns_car'].replace({'Yes' : 1, 'No' : 2}, inplace=True)

    return data


# drop irrelevant variables
def drop_irrelevant_variables(data=None):
    """ To drop irrelevant variables"""
    data.drop(['customer_id', 'first_name', 'last_name',
               'DOB','job_title', 'deceased_indicator', 'address', 'postcode', 'country'],
              axis=1, inplace=True)

    return data


# transform numerical and categorical variables.

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
