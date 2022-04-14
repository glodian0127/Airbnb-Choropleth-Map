import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer


K = 4
COLS = ['price', 'PopDens', 'TourEmRate', 'Mincome', 'Mvalue']


def load_data(path):
    data_raw = pd.read_excel(path, engine='openpyxl')
    return data_raw


def price_without_outlier(data):
    data_no = data[data.price <= 500].reset_index(drop=True)
    return data_no


def scale_data(data):
    # data_model = data.drop("id", 1).drop("host_id", 1).drop("FIPS", 1)  # drop ids' columns
    # data_model = data_model.drop('State', 1).drop('city', 1).drop('host_listings_count', 1).drop('room_type', 1)\
    #     .drop('set', 1)
    # features = list(data_model.columns)
    data_model = data[COLS]
    data_model[['Mincome', 'Mvalue']] = StandardScaler().fit_transform(data_model[['Mincome', 'Mvalue']])
    data_model[['PopDens']] = StandardScaler().fit_transform(data_model[['PopDens']])
    data_scaled = data_model
    return data_scaled


def find_optimal_k(data):
    kelbow_visualizer(KMeans(random_state=19), data, k=(1, 21))


def k_clustering(k, data, data_raw):
    kmeans = KMeans(n_clusters=k, random_state=19)
    label = kmeans.fit_predict(data)
    labels = np.array(list(label))
    data['latitude'] = data_raw['latitude']
    data['longitude'] = data_raw['longitude']
    data['label'] = pd.DataFrame(label)
    scatter_x = np.array(list(data['longitude']))
    scatter_y = np.array(list(data['latitude']))
    color_map = np.array(['#e31a1c', '#fe9929', '#74c476', '#3f007d', '#9ecae1', '#fed976'])
    data.plot(kind='scatter', x='longitude', y='latitude', label='label', c=color_map[labels], cmap=plt.get_cmap("jet"),
              alpha=0.4, figsize=(10, 7))
    color_dict = {0: '#e31a1c', 1: '#fe9929', 2: '#74c476', 3: '#3f007d', 4: '#9ecae1', 5: '#fed976'}
    fig, ax = plt.subplots()
    for l in np.unique(labels):
        ix = np.where(labels == l)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=color_dict[l], label=l)
    ax.legend()
    plt.savefig('clustering.png')
    plt.close()
    return label


def export_labeled_data(data, label):
    data['label'] = pd.DataFrame(label)
    data.to_csv('labeled_data_all.csv', index=False)


if __name__ == "__main__":
    data = load_data("data_prepared.xlsx")
    data_n_o = price_without_outlier(data)
    data_scaled = scale_data(data_n_o)
    # print(data_scaled)
    find_optimal_k(data_scaled)
    k_clustering(K, data_scaled, data_n_o)
    labels = k_clustering(K, data_scaled, data_n_o)
    print(labels)
    export_labeled_data(data_n_o, labels)