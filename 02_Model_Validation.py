import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

house_data = pd.read_csv('./train.csv')
print("================================= modeling ===================================")
# prediction target==y
y = house_data.SalePrice

# choosing predictors X
price_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr']

X = house_data[price_predictors]

# define model - What type of model will it be
my_model = DecisionTreeRegressor()

# fit_model - Capture patterns from provided data
my_model.fit(X, y)

print(my_model)

print("================================= prediction ===================================")
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
prediction_prices = my_model.predict(X)
print(prediction_prices)

print("================================= calculate error ===================================")
MAE = mean_absolute_error(y, prediction_prices)
print(MAE)

print("================================== resize the data set and run ==========================")
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, validate_X, train_y, validate_y = train_test_split(X, y, random_state=0)

my_decision_tree_model = DecisionTreeRegressor()
my_decision_tree_model.fit(train_X, train_y)

val_predictions = my_decision_tree_model.predict(validate_X)
print(mean_absolute_error(validate_y, val_predictions))


def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return (mae)


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, validate_X, train_y, validate_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))
