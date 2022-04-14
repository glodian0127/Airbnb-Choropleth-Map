import pandas as pd
import numpy as np


def get_city_name(path):
    data = pd.read_excel(path, engine='openpyxl')
    cities = data['city'].unique()
    return cities


if __name__ == "__main__":
    cities_name = get_city_name("data_prepared.xlsx")
    print(cities_name)