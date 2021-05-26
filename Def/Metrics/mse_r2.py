from sklearn.metrics import mean_squared_error, r2_score

def mse_r2(y_true, y_pred):
  
    #MSE
    print('Mean Squared Error: %.2f' % mean_squared_error(y_true,y_pred))

    #RSquared

    print('RSquared score: %.2f' % r2_score(y_true,y_pred))



def mse_r2_no_print(y_true, y_pred):

    return mean_squared_error(y_true,y_pred), r2_score(y_true,y_pred)
