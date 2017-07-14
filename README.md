# Yatra Airlines dataset

### Objective: Predict the Overall Rating

## Task 1:

1. Importing data
2. Data Wrangling
3. Missing Value imputation
4. Scaling
5. Encoding

### Write a function `import_data` that

#### Imports data from given path and select given list of columns:

For the given dataset, the columns to be selected:

* 'type_traveller'
* 'cabin_flown',
* 'overall_rating',
* 'seat_comfort_rating',
* 'cabin_staff_rating',
* 'food_beverages_rating',
* 'inflight_entertainment_rating',
* 'ground_service_rating',
* 'wifi_connectivity_rating',
* 'value_money_rating'

#### Drops necessary rows and columns:

If, missing values in any column are more than given threshold, then that column should be dropped by the function.

The threshold is defined as fraction of number of rows in the dataframe.

* e.g. threshold = 0.4 means if more than 40% of the values in a particular column is missing, then that column should be dropped.

Since we are trying to predict the `overall_rating`, the function should drop any rows with missing values in `overall_rating` columns

#### Imputes missing values:

For numerical data, use median.
For categoical data, use mode.

#### Centers and scales data:

#### Encodes categorical variables:

Apply appropriate encoding system.

Parameters:

* `path` : string
* `columns`: list of column names
* `encoding`: dict
* `drop_threshold` : float [0, 1], optional

Returns:

* `dataframe`: with selected columns and processed data
* `list`: of dropped column names above the threshold


## Task 2:

1. Train a linear regression algorithm
2. Perform grid search
3. Plot residual plots for each numeric variable

### Write a function `model_tuning` that

#### Trains a linear regression model
#### Performs Random Search
#### Plot residual plots for each numeric variable for the best model

Parameters:

* `DataFrame`
* `predictors` : list
* `target`: string
* `error metric`: string (optional)

Returns:

* `best_score`: float
* `params_to_best_moodel`: dict

Note: Include Error Handling