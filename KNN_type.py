import pandas as pd
import numpy as np
from plotly.offline import iplot
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go


def load_data(path):
    data_raw = pd.read_csv(path)
    return data_raw


def price_without_outlier(data):
    data_no = data[data.price <= 500].reset_index(drop=True)
    return data_no


def get_price_entireapt(data):
    data_EntireApt = data[data['Entire home/apt'] == 1]
    data_EntireApt = data_EntireApt.drop(['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'], 1)
    return data_EntireApt


def get_price_hotelroom(data):
    data_HotelRoom = data[data['Hotel room'] == 1]
    data_HotelRoom = data_HotelRoom.drop(['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'], 1)
    return data_HotelRoom


def get_price_privateroom(data):
    data_PrivateRoom = data[data['Private room'] == 1]
    data_PrivateRoom = data_PrivateRoom.drop(['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'], 1)
    return data_PrivateRoom


def get_price_sharedroom(data):
    data_SharedRoom = data[data['Shared room'] == 1]
    data_SharedRoom = data_SharedRoom.drop(['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'], 1)
    return data_SharedRoom


def scale_data(data):
    data_model = data.drop("id", 1).drop("host_id", 1).drop("fips", 1)  # drop ids' columns
    data_model_0 = data_model.drop("price", 1)
    features = list(data_model_0.columns)
    data_model[features] = StandardScaler().fit_transform(data_model[features])
    data_scaled = data_model
    return data_scaled


def split_data(data):
    data_x = data.drop("price", 1)
    data_y = data['price']
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=19)
    return x_train, x_test, y_train, y_test


def fit_knn_model_type(x_train, y_train, roomtype):
    k_range = np.arange(1, 31, 1)
    k_scores = []

    for k in k_range:
        price_knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        price_knn.fit(x_train, y_train)
        scores = cross_val_score(price_knn, x_train, y_train)
        # print(scores.mean())
        k_scores.append(scores.mean())

    mse = [1 - x for x in k_scores]
    trace = go.Scatter(y=mse, x=k_range, mode='lines+markers', marker=dict(color='rgb(150, 10, 10)'))
    layout = go.Layout(title='CV Error vs K_value ' + roomtype, xaxis=dict(title='K value', tickmode='linear'),
                       yaxis=dict(title='CV Error'))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig, filename='basic-line')
    opt_k = k_range[mse.index(min(mse))]
    return opt_k


def model_train_optimal_type(x_train, y_train, x_test, y_test, opt_k):
    knn_optimal = KNeighborsRegressor(n_neighbors=opt_k, weights='distance')
    knn_optimal.fit(x_train, y_train)
    y_pred = knn_optimal.predict(x_test)
    acc = knn_optimal.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    return acc, mse


def get_model_rmse(mse_type_1, mse_type_2, mse_type_3, data_1, data_2, data_3):
    n_data_1 = data_1.shape[0]
    n_data_2 = data_2.shape[0]
    n_data_3 = data_3.shape[0]
    mse_total = ((mse_type_1 * n_data_1) + (mse_type_2 * n_data_2) + (mse_type_3 * n_data_3)) / \
                (n_data_1 + n_data_2 + n_data_3)
    return mse_total


if __name__ == "__main__":
    data = load_data("cleanedData.csv")
    data_n_o = price_without_outlier(data)

    data_entireApt = get_price_entireapt(data_n_o)
    # data_hotelRoom = get_price_hotelroom(data_n_o)
    data_privateRoom = get_price_privateroom(data_n_o)
    data_sharedRoom = get_price_sharedroom(data_n_o)

    scaled_data_ea = scale_data(data_entireApt)
    # scaled_data_hr = scale_data(data_hotelRoom)
    scaled_data_pr = scale_data(data_privateRoom)
    scaled_data_sr = scale_data(data_sharedRoom)

    x_tr_ea, x_te_ea, y_tr_ea, y_te_ea = split_data(scaled_data_ea)
    # x_tr_hr, x_te_hr, y_tr_hr, y_te_hr = split_data(scaled_data_hr)
    x_tr_pr, x_te_pr, y_tr_pr, y_te_pr = split_data(scaled_data_pr)
    x_tr_sr, x_te_sr, y_tr_sr, y_te_sr = split_data(scaled_data_sr)

    optimal_k_ea = fit_knn_model_type(x_tr_ea, y_tr_ea, 'Entire_Apt')
    print("Optimal K of Entire Apt: " + str(optimal_k_ea))
    # optimal_k_hr = fit_knn_model_type(x_tr_hr, y_tr_hr, 3)
    # print("Optimal K of Hotel Room: " + optimal_k_hr)
    optimal_k_pr = fit_knn_model_type(x_tr_pr, y_tr_pr, 'Private Room')
    print("Optimal K of Private Room: " + str(optimal_k_pr))
    optimal_k_sr = fit_knn_model_type(x_tr_sr, y_tr_sr, 'Shared Room')
    print("Optimal K of Shared Room: " + str(optimal_k_sr))

    accuracy_ea, mse_ea = model_train_optimal_type(x_tr_ea, y_tr_ea, x_te_ea, y_te_ea, optimal_k_ea)
    print("Accuracy of Entire Apt: " + str(accuracy_ea))
    print("MSE of Entire Apt: " + str(mse_ea))
    # accuracy_hr, mse_hr = model_train_optimal_type(x_tr_hr, y_tr_hr, x_te_hr, y_te_hr, optimal_k_hr)
    # print("Accuracy of Hotel Room: " + str(accuracy_hr))
    # print("MSE of Entire Hotel Room: " + str(mse_hr))
    accuracy_pr, mse_pr = model_train_optimal_type(x_tr_pr, y_tr_pr, x_te_pr, y_te_pr, optimal_k_pr)
    print("Accuracy of Private Room: " + str(accuracy_pr))
    print("MSE of Entire Private Room: " + str(mse_pr))
    accuracy_sr, mse_sr = model_train_optimal_type(x_tr_sr, y_tr_sr, x_te_sr, y_te_sr, optimal_k_sr)
    print("Accuracy of Shared Room: " + str(accuracy_sr))
    print("MSE of Entire Shared Room: " + str(mse_sr))
    mse_Type = get_model_rmse(mse_ea, mse_pr, mse_sr, data_entireApt, data_privateRoom,
                              data_sharedRoom)
    print("MSE: " + str(mse_Type))