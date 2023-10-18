import numpy as np
from numpy import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def cos_similarity(arr1, arr2):
    arr1_norm = np.linalg.norm(arr1)
    arr2_norm = np.linalg.norm(arr2)
    if arr1_norm == 0 and arr2_norm == 0:
        return 1
    elif arr1_norm == 0 or arr2_norm == 0:
        return 0
    else:
        return np.dot(arr1, arr2) / (arr1_norm * arr2_norm)


def another_similarity(arr1, arr2):
    a1 = np.array(arr1)
    a2 = np.array(arr2)
    sub = a1 - a2
    multi = np.multiply(sub, sub.T)
    if np.linalg.norm(multi) == 0:
        return 1
    else:
        return 1 / sqrt(np.linalg.norm(multi))


def simply_str_num(x: str):
    x_list = x.split('.')
    try:
        ret = x_list[0] + '.' + x_list[1][:5]
        return ret
    except Exception as ex:
        print('simply_str_num... Exception: ' + str(ex))


def get_evaluate_result(predict_arr, real_arr, epsilon=1):
    """
    获取评价指标结果
    返回 MAE, MSE, RMSE, MAPE
    """
    MSE = mean_squared_error(real_arr, predict_arr)
    MAE = mean_absolute_error(real_arr, predict_arr)
    RMSE = sqrt(MSE)
    MAPE = np.mean(np.abs(predict_arr - real_arr) / (real_arr + epsilon))
    r2 = r2_score(real_arr, predict_arr)
    return MAE, MSE, RMSE, MAPE, r2


if __name__ == '__main__':
    another_similarity([1,2,3], [2,3,3])