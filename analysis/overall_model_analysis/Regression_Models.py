# Basic code for Regression - RV

## Basic LR

# Select label and features from temp_by_city dataframe
label = temp_by_city_encoded['Avg_temp'].dropna()
# feature_list = ['City', 'Country', 'Continent', 'Lat_num', 'Long_num', 'Year', 'Month', 'Avg_temp_uncertainty']
feature_list = []
features = temp_by_city_encoded[feature_list].dropna()

# split data into train/ test 
seed = 42
x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split(features, label, test_size = 0.2, random_state = seed)

lr_model = LinearRegression().fit(x_train_lr, y_train_lr)
pred_train_lr= lr_model.predict(x_train_lr)
pred_test_lr = lr_model.predict(x_test_lr)

# # calculate accuracy for train
# print(f'Train R2 score: {r2_score(y_train_lr.values, pred_train_lr)}')
# print(f'Train RMSE score: {np.sqrt(mean_squared_error(y_train_lr,pred_train_lr))}')

# calculate accuracy for test
print(f'Test RMSE score: {np.sqrt(mean_squared_error(y_test_lr.values, pred_test_lr))}')
print(f'Test R2 score: {r2_score(y_test_lr.values, pred_test_lr)}')


## Ridge 

rr = Ridge(alpha=10)
rr.fit(x_train_lr, y_train_lr) 

# pred_train_rr= rr.predict(x_train_lr)
# print(f'Train RMSE score: {np.sqrt(mean_squared_error(y_train_lr,pred_train_rr))}')
# print(f'Train R2 score: {r2_score(y_train_lr, pred_train_rr)}')

pred_test_rr= rr.predict(x_test_lr)
print(f'Test RMSE score: {np.sqrt(mean_squared_error(y_test_lr,pred_test_rr))}') 
print(f'Test R2 score: {r2_score(y_test_lr, pred_test_rr)}')

# Optimizing Ridge 
lambdas = np.linspace(1, 100, 1)

ridge_cv = RidgeCV(alphas=lambdas, scoring='neg_mean_squared_error')
ridge_cv.fit(x_train_lr, y_train_lr)
ideal_alpha = ridge_cv.alpha_

rr = Ridge(alpha=ideal_alpha)
rr.fit(x_train_lr, y_train_lr) 

# pred_train_rr= rr.predict(x_train_lr)
# print(f'Train RMSE score: {np.sqrt(mean_squared_error(y_train_lr,pred_train_rr))}')
# print(f'Train R2 score: {r2_score(y_train_lr, pred_train_rr)}')

pred_test_rr= rr.predict(x_test_lr)
print(f'Test RMSE score: {np.sqrt(mean_squared_error(y_test_lr,pred_test_rr))}') 
print(f'Test R2 score: {r2_score(y_test_lr, pred_test_rr)}')

## LASSO 
model_lasso = Lasso(alpha = 10)
model_lasso.fit(x_train_lr, y_train_lr) 

# pred_train_lasso= model_lasso.predict(x_train_lr)
# print(f'Train RMSE score: {np.sqrt(mean_squared_error(y_train_lr,pred_train_lasso))}')
# print(f'Train R2 score: {r2_score(y_train_lr, pred_train_lasso)}')

pred_test_lasso= model_lasso.predict(x_test_lr)
print(f'Test RMSE score: {np.sqrt(mean_squared_error(y_test_lr,pred_test_lasso))}') 
print(f'Test R2 score: {r2_score(y_test_lr, pred_test_lasso)}')

# optimizing LASSO 
lambdas = np.linspace(1, 100, 1)

lasso_cv = LassoCV(alphas=lambdas)
lasso_cv.fit(x_train_lr, y_train_lr)
ideal_alpha = lasso_cv.alpha_

model_lasso = Lasso(alpha=ideal_alpha)
model_lasso.fit(x_train_lr, y_train_lr) 

# pred_train_lasso= model_lasso.predict(x_train_lr)
# print(f'Train RMSE score: {np.sqrt(mean_squared_error(y_train_lr,pred_train_lasso))}')
# print(f'Train R2 score: {r2_score(y_train_lr, pred_train_lasso)}')

pred_test_lasso= model_lasso.predict(x_test_lr)
print(f'Test RMSE score: {np.sqrt(mean_squared_error(y_test_lr,pred_test_lasso))}') 
print(f'Test R2 score: {r2_score(y_test_lr, pred_test_lasso)}')

## ElasticNet
model_elastic = ElasticNet(alpha = 10)
model_elastic.fit(x_train_lr, y_train_lr) 

# pred_train_elastic= model_elastic.predict(x_train_lr)
# print(f'Train RMSE score: {np.sqrt(mean_squared_error(y_train_lr, pred_train_elastic))}')
# print(f'Train R2 score: {r2_score(y_train_lr, pred_train_elastic)}')

pred_test_elastic= model_elastic.predict(x_test_lr)
print(f'Test RMSE score: {np.sqrt(mean_squared_error(y_test_lr, pred_test_elastic))}') 
print(f'Test R2 score: {r2_score(y_test_lr, pred_test_elastic)}')

## optimizing ElasticNet
lambdas = np.linspace(1, 100, 1)
ratios = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1]

elastic_cv = ElasticNetCV(l1_ratio = ratios, alphas=lambdas)
elastic_cv.fit(x_train_lr, y_train_lr)
ideal_alpha = elastic_cv.alpha_
ideal_ratio = elastic_cv.l1_ratio_

model_elastic = ElasticNet(l1_ratio = ideal_ratio, alpha = ideal_alpha)
model_elastic.fit(x_train_lr, y_train_lr) 

# pred_train_elastic= model_elastic.predict(x_train_lr)
# print(f'Train RMSE score: {np.sqrt(mean_squared_error(y_train_lr, pred_train_elastic))}')
# print(f'Train R2 score: {r2_score(y_train_lr, pred_train_elastic)}')

pred_test_elastic= model_elastic.predict(x_test_lr)
print(f'Test RMSE score: {np.sqrt(mean_squared_error(y_test_lr, pred_test_elastic))}') 
print(f'Test R2 score: {r2_score(y_test_lr, pred_test_elastic)}')