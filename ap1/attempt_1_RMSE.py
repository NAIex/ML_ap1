import os
import matplotlib.pyplot as plt
import numpy as np

from id3 import ID3
from data import *

def section(e , d):
    return np.array(list(filter(e,d)))

months = {1,2,3,4,5,6,7,8,9,10,11,12}

d2021_2022_2023, _ = load_data([DATA_2021,DATA_2022,DATA_2023], months)
dlong, _ = load_data([DATA_2023], months)
dfull, n = load_data([DATA_2022,DATA_2023], months)
d2024, _ = load_data([DATA_2024], months)
d2023, _ = load_data([DATA_2023], months)
d2023_2024, _ = load_data([DATA_2023,DATA_2024],{1,2,3,4,5,6,7,8,9,10,11})
d2024_dec, _ = load_data([DATA_2024_DEC],{12})

d2019_2021_2022, _ = load_data([DATA_2019,DATA_2021,DATA_2022],months)


predict_month = 12

dtest = section(lambda x: x[1] in {predict_month}, d2024_dec)
dyears = section(lambda x: x[1] in {predict_month}, d2023)
dmonths2 = section(lambda x: x[1] in {predict_month-1}, d2023_2024)
dmonths4 = section(lambda x: x[1] in {predict_month-1}, dfull)
dyears_months2 = section(lambda x: x[1] in {predict_month-1}, d2023_2024)


nuclear        = extract_data_and_label(dyears,2,9)
consum         = extract_data_and_label(dyears,2,3)

hidrocarburi   = extract_data_and_label(section(lambda x: x[1] in {predict_month},d2021_2022_2023),2,7)
foto           = extract_data_and_label(section(lambda x: x[1] in {predict_month}, d2021_2022_2023),2,11)

ape            = extract_data_and_label(section(lambda x: x[1] in {predict_month-1},d2024),2,8)


carbune        = extract_data_and_label(dyears_months2,2,6)
eolian         = extract_data_and_label(dmonths2,2,10)
biomas         = extract_data_and_label(dmonths2,2,12)


split_size = 5

consum_t        = ID3(consum,30) 
nuclear_t       = ID3(nuclear,30)
hidrocarburi_t  = ID3(hidrocarburi,30)
ape_t           = ID3(ape,30)

foto_t          = ID3(foto,3)


eolian_t        = ID3(eolian,14)

carbune_t       = ID3(carbune,7)
biomas_t        = ID3(biomas,7)

full_pred, full_sum = 0,0

#print(d.shape)
split_pred = np.zeros(11)
split_real = np.zeros(11)

diff_sum = np.zeros(11)
for i in range(dtest.shape[0]):
    sum = 0
    date = dtest[i][2]

    #print()
    #print(date)

    '''
    
    v = consum_t.predict([date])
    #print(f'Consum{dtest[i][3]} Predict{v}')
    sum += v
    v = carbune_t.predict([date])
    #print(f'Carbune{dtest[i][6]} Predict{v}')
    sum -= v
    v= hidrocarburi_t.predict([date])
    #print(f'Hidrocarburi{dtest[i][7]} Predict{v}')
    sum -= v
    v= ape_t.predict([date])
    #print(f'Ape{dtest[i][8]} Predict{v}')
    sum -= v
    v= nuclear_t.predict([date])
    #print(f'Nuclear{dtest[i][9]} Predict{v}')
    sum -= v
    v= eolian_t.predict([date])
    #print(f'Eolian{dtest[i][10]} Predict{v}')
    sum -= v
    v= foto_t.predict([date])
    #print(f'Foto{dtest[i][11]} Predict{v}')
    sum -= v
    v= biomas_t .predict([date])
    #print(f'Biomas{dtest[i][12]} Predict{v}')
    sum -=v
    #print(d[i][0])
    #print(f'{d[i][11]}<-Real\n{sum}<-Prediction\n')
    
    full_pred += sum
    full_sum += dtest[i][13]

    '''
    '''
    split_pred += [
        consum_t.predict([date]),
        0,
        0,
        carbune_t.predict([date]),
        hidrocarburi_t.predict([date]),
        ape_t.predict([date]),
        nuclear_t.predict([date]),
        eolian_t.predict([date]),
        foto_t.predict([date]),
        biomas_t.predict([date]),
        0]
    split_real += dtest[i][3:]
    '''

    diff_sum += np.pow(np.array([
        consum_t.predict([date]),
        0,
        0,
        carbune_t.predict([date]),
        hidrocarburi_t.predict([date]),
        ape_t.predict([date]),
        nuclear_t.predict([date]),
        eolian_t.predict([date]),
        foto_t.predict([date]),
        biomas_t.predict([date]),
        0]) - dtest[i][3:],2)

diff_sum /= dtest.shape[0]
diff_sum = np.sqrt(diff_sum)

n = (n.flatten())[1:]

for label, percentage in zip(n,diff_sum):
    print(f'{label} => {percentage}')

'''
res = np.array((split_real / split_pred) * 100, dtype='float')
n = n.flatten()
n = n[1:]
for label, percentage in zip(n,res):
    print(f'{label} => {percentage}')
print('\nPREDICTION',full_pred,full_sum, sep='\n')

print('Loaded data set')
'''

