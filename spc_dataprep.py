# create customers table
def create_customer_table(demographics, address):
    """ Concatenates demographics table and address table using only the 2nd to 6th columns of address"""
    customers_table = pd.concat([demographics, address.iloc[:, 1:6]], axis=1)

    return customers_table


# keeps on approved transactions
def approved_transactions(transactions):
    """ Keeps only approved transactions from the transactions table  and returs
    data with customer_id, list_price, standard_cost and order_status variables. """

    to_keep = ['customer_id', 'list_price', 'standard_cost', 'order_status']
    transactions_approved = transactions.loc[trans_cop['order_status']=='Approved', to_keep]

    return transactions_approved


# creates profit column in transactions table
def create_profit_variable(transactions):
    """ Creates a profit variable by subtracting standard_cost from list_price. """
    transactions['profit'] = transactions['list_price'] - transactions['standard_cost']

    return transactions


# creates profit column inside customers table
def customers_with_profit(customers_table, transactions_table):
    """ Adds up profit of each customer_id from the transaction table to the customers table"""
    profit = []
    for ID in customers_table['customer_id']:
        prof = transactions_table.loc[transactions_table['customer_id'] == ID, 'profit'].sum()
        profit.append(prof)

    customers_table['profit'] = profit

    return customers_table


# splits data into X and y
def x_y_data_split(data):
    """ Splits data to predictors and target"""
    X = data.drop(['profit'], axis=1)
    y = data['profit']

    return X, y
