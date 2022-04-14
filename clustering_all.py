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


def pie_chart_cities(data_raw):
    category = data_raw['label'].unique().tolist()

    labels = category
    sizes = data_raw['label'].value_counts().sort_index().tolist()
    data = [i for i in sizes]
    data = np.array(data)
    explodes = np.zeros(len(data))
    explodes[data.argmax()] = 0.1
    plt.pie(data, explode=explodes, labels=labels, autopct='%1.2f%%', pctdistance=0.6, labeldistance=1.2,
                shadow=True)
    plt.title('all_city')
    plt.legend()
    plt.savefig('all.png')
    plt.close()
    print(sizes)


if __name__ == "__main__":
    data = load_data("labeled_data_all.csv")
    cities = city_names(data)
    # print(cities)
    pie_chart_cities(data)
