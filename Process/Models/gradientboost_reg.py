from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import Implementation.results as rs

def gradientBoosting(x, y, n_estimators):

    X_train, X_test, Y_train, Y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=33,
    )

    gradientBoostingRegressor = GradientBoostingRegressor(
        loss="ls",
        learning_rate=0.1,
        n_estimators=n_estimators,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=6,
        min_impurity_decrease=0.0,
        init=None,
        random_state=12345,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        presort="auto",
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=0.0001,
    )

    gradientBoostingRegressor.fit(X_train, Y_train)


    Y_pred_Boost = gradientBoostingRegressor.predict(X_test)
    
    return Y_test, Y_pred_Boost
    #rs.results(Y_test, Y_pred_Boost)