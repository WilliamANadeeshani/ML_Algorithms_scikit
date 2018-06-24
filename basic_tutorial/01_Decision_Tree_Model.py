import pandas as pd
from sklearn.tree import DecisionTreeRegressor

house_data = pd.read_csv('./Data/train.csv')

print("================================= indexing ===================================")

# print columns
print(house_data.columns)

# Use the head command to print first lines the top few lines of the price
print(house_data.SalePrice.head())

# Pick any two variables and store them to a new DataFrame use describe to summarize the data
heatData = ['Heating', 'HeatingQC']
print (house_data[heatData].describe())

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
print(my_model.predict(X.head()))



