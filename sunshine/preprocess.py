from sklearn.ensemble import RandomForestRegressor


__all__ = ['trans_biz_type']
# 转换web、app为数值
def trans_biz_type(x):
    ret = x
    if x == 'web':
        ret = 1
    elif x == 'app':
        ret = 2
    return ret

# 对level缺失值用随机森林的方法进行填充--训练集
# 创建随机森林函数
def fill_by_RandomForest(data, pred_field):
    known = data[data[pred_field].notnull()]
    unknown = data[data[pred_field].isnull()]
    x_train = known.iloc[:, [2, 4, 5, 9, 10]]
    y_train = known.loc[:, 'level']
    x_test = unknown.iloc[:, [2, 4, 5, 9, 10]]
    rfr = RandomForestRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=5)
    pred_y = rfr.fit(x_train, y_train).predict(x_test)
    return pred_y