
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,SGDRegressor,ElasticNet,Lars,Lasso,OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor,RandomTreesEmbedding

def datamodels(models,train_split):
    for name,model in models.items():
        print(f"--------Training:{name}--------")
        model=model.fit(train_split[0],train_split[2])
        y_predict=model.predict(train_split[1])
        print(f"RMSE->{mean_squared_error(y_predict,train_split[3])**0.5}")

