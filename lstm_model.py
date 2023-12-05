

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 01:37:14 2023

@author: 35003
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

#####################################################################################
#####################################################################################
###parameters
#test GPU avaiable
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
###read data
path = os.getcwd()
raw=pd.read_csv(path+'\\'+'CAD_15min.csv')
# 定义日期范围
start_date = pd.to_datetime('2023-03-01 00:00:00')
end_date = pd.to_datetime('2023-03-24 23:59:00')
trade_start_date = pd.to_datetime('2023-03-25 00:00:00')
trade_end_date=pd.to_datetime('2023-03-31 16:00:00')

###model
lossrate=0.2
time_step = 96#4*24
batchsize=32
epoch=250
#forcast timezone
forcast_time=12

#######################################################################################
#######################################################################################
#data

# 将 "Dates" 列转换为 datetime 类型
raw['Dates'] = pd.to_datetime(raw['Dates'])

# 根据条件选取所需行
data_set = raw.loc[(raw['Dates'] >= start_date) & (raw['Dates'] <= end_date)]

# 重置 dataset 的索引
data_set.reset_index(drop=True, inplace=True)

vol = data_set.reset_index()['Close']

#plot vol
plt.figure(figsize=(14, 9))
plt.plot(vol)
plt.xlabel('Index')
plt.ylabel('Close')
plt.show()

###缩放data，scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
vol = scaler.fit_transform(np.array(vol).reshape(-1,1))
# 使用np.array(vol).reshape(-1,1)将vol转换为NumPy数组，并使用reshape()方法将其转换为一个二维数组

#distrribute train and test data
training_data_size = int(len(vol)*0.8)
test_data_size = len(vol)-training_data_size
train_data = vol[0:training_data_size,:]
test_data = vol[training_data_size:len(vol),:1]

#Converting an array of values into a dataset matrix.
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]    
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
# =============================================================================
# 这个函数create_dataset()用于将给定的数据集（如时间序列数据）转换为监督学习问题的输入（X）和输出（Y）。它需要两个参数：数据集（dataset）和时间步长（time_step，默认值为1）。
# 
# 输入数据集应为NumPy数组，其中每行表示一个观察值，每列表示一个特征。该函数将根据给定的时间步长，从数据集中提取输入（X）和输出（Y）序列。
# 
# 在函数中：
# 
# 定义两个空列表dataX和dataY来存储输入和输出序列。
# 使用for循环迭代数据集的元素。循环范围从0到len(dataset) - time_step - 1。
# 对于每个元素，从当前索引位置开始，提取与时间步长长度相等的子序列。将子序列添加到dataX列表中。
# 将与当前输入子序列对应的下一个观测值添加到dataY列表中。
# 将dataX和dataY列表转换为NumPy数组并返回。
# 这个函数对于准备基于时间窗口的预测任务（例如，时间序列预测）特别有用，因为它会自动为您创建所需的输入输出对。
# 
# =============================================================================

#Reshaping into X=t,t+1,t+2,t+3 (independent values) and Y=t+4.

#training set, test set
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
#input train/test
#output train/test

#X_train.shape, y_train.shape
#X_test.shape, y_test.shape

#Reshaping for LSTM Model input to be [samples, time steps, features].
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
# =============================================================================
# 这两行代码将X_train和X_test数组重塑为一个三维数组。
# .shape[]属性返回一个表示数组或DataFrame的形状的元组。.shape[0]表示数组或DataFrame的第一个维度的大小，即行数.[1]为列数
# X_train.shape[0]：训练集中的样本数。
# X_train.shape[1]：每个样本的时间步长（也就是观察值的个数）。
#每个时间步长的特征数。在这个例子中，只有一个特征（即股票价格或其他数值）。
# =============================================================================


#######################################################################################
##############################################################
#
###train model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout

def fit_model(time_step,lossrate,X_train, y_train, X_test, y_test, epoch,batchsize):
    model_vol = Sequential()
    model_vol.add(LSTM(time_step,return_sequences=True,input_shape=(time_step,1)))
    model_vol.add(Dropout(lossrate))
    model_vol.add(LSTM(time_step,return_sequences=True))
    model_vol.add(Dropout(lossrate))
    model_vol.add(LSTM(time_step))
    model_vol.add(Dropout(lossrate))
    model_vol.add(Dense(1))
    model_vol.compile(loss='mean_squared_error',optimizer='adam')
    loss_history = model_vol.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=batchsize, verbose=1)

#fit_model(time_step,lossrate,X_train, y_train, X_test, y_test, epoch,batchsize)

'''sl model'''
# model_vol.save(path+'\\'+"CAD_15min.h5")
model_vol = load_model(path+'\\'+"CAD_15min.h5")
model_vol.summary()


#plot loss function
def loss_function():
    import h5py
    with h5py.File("loss_history.h5", "w") as file:
        # 将训练集损失保存到h5文件
        file.create_dataset("CAD-price-train_loss", data=loss_history.history["loss"])
        # 将验证集损失保存到h5文件
        file.create_dataset("CAD-price-val_loss", data=loss_history.history["val_loss"])
        
def loss_function_plot():
    import h5py
    with h5py.File("loss_history.h5", "r") as file:
        train_loss = file["CAD-price-train_loss"][:]
        val_loss = file["CAD-price-val_loss"][:]
    
    # 获取损失值
    training_loss = loss_history.history['loss']
    validation_loss = loss_history.history['val_loss']
    
    # 获取epoch数量
    epochs = range(1, len(training_loss) + 1)
    
    # # 绘制损失曲线
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
#loss_function()
#loss_function_plot()


#######################################################################################
#####################################################
#measurement
train_predict = model_vol.predict(X_train)
test_predict = model_vol.predict(X_test)

train_predict_stat = pd.Series(model_vol.predict(X_train).squeeze())
test_predict_stat = pd.Series(model_vol.predict(X_test).squeeze())
y_train_stat=pd.Series(y_train)
y_test_stat=pd.Series(y_test)


def loglize(trans):
    # Convert the NumPy array to a pandas Series
    trans = trans.diff()


    # Check for NaN or infinite values and replace them with 0
    trans.fillna(0, inplace=True)

    # Add a 0 at the beginning of the Series
    trans = pd.concat([pd.Series([0]), trans], ignore_index=True)

    # Convert the Series to a 2D NumPy array
    trans = np.array(trans.values).reshape(-1, 1)

    # Replace any remaining NaN or infinite values with 0
    trans = np.where(np.isnan(trans) | np.isinf(trans), 0, trans)
    # 绘制 ACF 图
    plt.figure(figsize=(12, 4))
    plot_acf(trans, lags=10)
    plt.xlabel("Lags")
    plt.ylabel("ACF")
    plt.title("Autocorrelation Function")
    plt.show()

    # 绘制 PACF 图
    plt.figure(figsize=(12, 4))
    plot_pacf(trans, lags=10)
    plt.xlabel("Lags")
    plt.ylabel("PACF")
    plt.title("Partial Autocorrelation Function")
    plt.show()

    #  adf
    adf_result = adfuller(trans)

    # Print the test results
    print(f"ADF Statistic (differenced data): {adf_result[0]}")
    print(f"p-value (differenced data): {adf_result[1]}")

    # Check if the differenced time series is stationary
    if adf_result[1] < 0.05:
        print("The differenced time series is stationary.")
    else:
        print("The differenced time series is still not stationary.")

    return trans
    

# Apply the loglize transformation
variables = ['train_predict_stat', 'test_predict_stat', 'y_train_stat', 'y_test_stat']

for i in variables:
    globals()[i] = loglize(trans=globals()[i])


###########################################################################
#Calculating RMSE (Root mean squared error) of y_train and y_test
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print("training_set rmse:", math.sqrt(mean_squared_error(y_train_stat,train_predict_stat)))
print("test_set rmse:", math.sqrt(mean_squared_error(y_test_stat,test_predict_stat)))
# 计算训练集的 MAE
train_mae = mean_absolute_error(y_train_stat, train_predict_stat)
test_mae = mean_absolute_error(y_test_stat, test_predict_stat)
print("training_set MAE:", train_mae)
print("test_set MAE:", test_mae)

################################################################
################################################################
#Shifting the train and test predictions for plotting.

#anti-scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

look_back = time_step
trainPredictPlot = np.empty_like(vol)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(vol)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(vol)-1, :] = test_predict

plt.figure(figsize=(16,10))# 使用 plt.figure() 创建一个新的绘图窗口，并设置窗口的大小为 16x10 英寸。
plt.plot(scaler.inverse_transform(vol))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# =============================================================================
# look_back 是用于创建数据集的时间步长，这里设置为50。
# trainPredictPlot 和 testPredictPlot 是两个空的 NumPy 数组，其形状与原始股票价格序列 vol 相同。它们将被用于存储训练集和测试集的预测结果。
# 接下来的几行代码将 train_predict 和 test_predict 中的预测结果插入到 trainPredictPlot 和 testPredictPlot 的相应位置。注意，预测结果的起始点是根据 look_back 的值进行偏移的，以确保预测与原始数据的时间对齐。
# =============================================================================




################################################################################
################################################################################
'''trade price forecast###'''
#end_date
#trade_end_date

trade_data_set = raw.loc[(raw['Dates'] >= end_date) & (raw['Dates'] <= trade_end_date)]
len(trade_data_set)
type(trade_data_set)
trade_data_close= trade_data_set['Close']
trade_data_mean=pd.DataFrame({})
trade_data_mean[:]= trade_data_close.mean()

from statsmodels.tsa.arima.model import ARIMA



# 拟合ARIMA(1,1,1)模型
arima_model = ARIMA(trade_data_close, order=(1, 1, 1))
arima_result = arima_model.fit()

# 获取拟合后的差分数据
fitted_values_diff = arima_result.fittedvalues
fitted_values_diff_from_second = fitted_values_diff.iloc[1:]

# 将trade_data_close的第一位接上fitted_values_diff_from_second
fitted_values_diff_cumsum = pd.concat([trade_data_close[:1], fitted_values_diff_from_second])
fitted_values = fitted_values_diff_cumsum
# 计算移动平均（例如，采用窗口大小为5的简单移动平均）
window_size = 5
moving_average = trade_data_close.rolling(window=window_size).mean()
moving_average[0:5]=trade_data_close[0:5]

#lstm
trade_data_set_2d = np.array(trade_data_close).reshape(-1, 1)
#len(test_data[-time_step:])
#len(trade_data_set_2d)
# 使用concatenate()函数沿着第一轴（axis=0）拼接两个数组
trade_true_data = np.concatenate([scaler.inverse_transform(test_data[-time_step:]), trade_data_set_2d], axis=0)


x_input=trade_true_data.reshape(1,-1)#从某一天的第一分钟开始
x_input.shape#=n_steps+1
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
#len(temp_input)
# 输出列表的前 5 个元素,检查以确保列表被正确创建并包含期望的数据。
temp_input[:5]
# =============================================================================
# x_input = test_data[39:].reshape(1, -1)：从 test_data 中提取最后几个元素（从索引39开始到最后一个元素），然后使用 reshape(1, -1) 将其转换为一个二维数组，其中第一个维度是 1，表示一个数据样本，第二个维度是由数据元素数自动推断的。
# x_input.shape：输出 x_input 的形状。这里应该是一个形状为 (1, n) 的二维数组，其中 n 是从测试数据中提取的元素个数。
# temp_input = list(x_input)：将 x_input 转换为 Python 列表。注意，这会创建一个嵌套的列表，因为 x_input 是一个二维数组。
# temp_input = temp_input[0].tolist()：从嵌套列表中提取第一个元素（即 x_input 中的一维数组），并将其转换为 Python 列表。此时，temp_input 是一个包含从测试数据中提取的元素的一维列表。
# =============================================================================
temp_input=scaler.fit_transform(np.array(temp_input).reshape(-1,1))
np.shape(temp_input)

#type(temp_input)
lst_output=[]
n_steps=time_step
i=0


# 开始循环
for i in range(len(temp_input) - time_step):
    # 选取从第i个开始的40个数据作为输入
    input_data = temp_input[i : i + time_step]
    
    # 将输入数据调整为LSTM模型所需的形状（批量大小, 时间步长, 特征数）
    input_data = input_data.reshape((1, time_step, -1))
    
    # 预测下一个数据
    yhat = model_vol.predict(input_data)
    
    # 将预测值添加到输出列表中
    lst_output.append(yhat[0])

inverse_output = np.array(scaler.inverse_transform(lst_output))
len(inverse_output)
inverse_output=pd.Series(inverse_output[:,0])
inverse_output.index=moving_average.index

#plot#######################################################
# 绘制实际价格与预测价格
plt.figure(figsize=(12, 6))
plt.plot(trade_data_close, label="Actual Price")
plt.plot(inverse_output, label='forecast')
plt.title('actual price vs lstm forecasted price')
plt.legend()
plt.show()

# 绘制实际价格与预测价格
plt.figure(figsize=(12, 6))
plt.plot(trade_data_close, label="Actual Price")
plt.plot(moving_average, color="green", label=f"{window_size}-Day Moving Average")
plt.legend()
plt.title('actual price vs SMA forecasted price')
plt.show()


# 绘制实际价格与预测价格
plt.figure(figsize=(12, 6))
plt.plot(trade_data_close, label="Actual Price")
plt.plot(fitted_values, color="red", label="Fitted Values")
plt.legend()
plt.title('actual price vs ARIMA(1,1,1) forecasted price')
plt.show()

###key stat################################################################
#RMSE
rmse_arima = np.sqrt(mean_squared_error(trade_data_close, fitted_values))
rmse_sma = np.sqrt(mean_squared_error(trade_data_close, moving_average.dropna()))
rmse_lstm = np.sqrt(mean_squared_error(trade_data_close, inverse_output))

trade_data_close.isna().any()
fitted_values.isna().any()
moving_average.isna().any()
print("ARIMA RMSE:", rmse_arima)
print("SMA RMSE:", rmse_sma)
print("LSTM RMSE:", rmse_lstm)

# 计算ARIMA、SMA和LSTM预测价格的MAE
mae_arima = mean_absolute_error(trade_data_close, fitted_values)
mae_sma = mean_absolute_error(trade_data_close, moving_average)
mae_lstm = mean_absolute_error(trade_data_close, inverse_output)

print("ARIMA MAE:", mae_arima)
print("SMA MAE:", mae_sma)
print("LSTM MAE:", mae_lstm)


#########################################################################
'''save forecasted price#'''
# 将DataFrame对象保存为CSV文件
result_data_set= trade_data_set[['Dates','Close']].reset_index()

result_data_set['Forecasted_Close'] = inverse_output
result_data_set = result_data_set[['Dates', 'Forecasted_Close','Close']]
result_data_set.to_csv(path+'\\'+'CAD-forecasted price-15min of 1M.csv', index=False)













