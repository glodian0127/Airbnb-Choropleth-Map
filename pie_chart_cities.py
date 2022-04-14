import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data_raw = pd.read_csv(path)
    return data_raw


def city_names(data_raw):
    cities = data_raw['city'].unique()
    data_city = cities.tolist()
    return data_city


def pie_chart_cities(data_raw, cities):
    category = data_raw['label'].unique().tolist()

    for city in cities:
        data_city = data_raw.loc[data_raw['city'] == city]
        labels = category
        sizes = data_city['label'].value_counts().sort_index().tolist()
        data = [i for i in sizes]
        data = np.array(data)
        explodes = np.zeros(len(data))
        explodes[data.argmax()] = 0.1
        plt.pie(data, explode=explodes, labels=labels,
                autopct=lambda p: '{:.2f}%({:.0f})'.format(p, (p/100)*data.sum()), pctdistance=0.6,
                labeldistance=1.2, shadow=True)
        plt.title(city)
        plt.legend()
        plt.savefig(city + '_Pie_Chart.png')
        plt.close()


if __name__ == "__main__":
    data = load_data("labeled_data_all.csv")
    cities = city_names(data)
    pie_chart_cities(data, cities)
