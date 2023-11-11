"""
Authors: Magdalena Asmus-Mrzygłów, Patryk Klimek
Function used to generate JSON file with movies preferences data form Excel file
"""
import json

import pandas as pd
# pip install openpyxl


def create_data_json():
    """
    Generating JSON file with movies recommendation data from Excel file
    """
    movie_data = pd.read_excel('movies_data.xlsx')
    data_dict = {}

    # iterate through columns
    for col_num in range(len(movie_data.columns)):
        movies_score = {}
        #iterate through movie rows
        for num in range(0, len(movie_data.iloc[:, col_num]), 2):
            # create movies:score dictionary
            if str(movie_data.iloc[num, col_num]) != 'nan':
                movies_score.update({movie_data.iloc[num, col_num]: movie_data.iloc[num + 1, col_num]})

        data_dict.update({movie_data.columns[col_num]: movies_score})

    # create json object from dictionary with ensuring correct encoding of Polish characters
    movies_json = json.dumps(data_dict, ensure_ascii=False, indent=2)

    # save json object to file
    with open('movies_data.json', 'w+', encoding='utf-8') as file:
        file.write(movies_json)

# execute function
create_data_json()
