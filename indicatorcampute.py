

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error




class computeindicator():
    def __init__(self, tar, pre):
        self.sum_rmse = []
        self.sum_mae = []
        self.sum_mape = []
        self.tar = tar
        self.pre = pre

    def compute(self):
        for i in range(self.tar.shape[1]):
            rmse = np.sqrt(mean_squared_error(self.pre[i], self.tar[i]))
            self.sum_rmse.append(rmse)

            mae = mean_absolute_error(self.pre[i], self.tar[i])
            self.sum_mae.append(mae)
            mape = mean_absolute_percentage_error(self.pre[i], self.tar[i])
            self.sum_mape.append(mape)

        mean_rmse = np.mean(np.array(self.sum_rmse))
        mean_mae = np.mean(np.array(self.sum_mae))
        print('mean_mae ={:5.6f} mean_rmse ={:5.6f}'.format(mean_mae, mean_rmse))
