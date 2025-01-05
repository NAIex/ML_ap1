import os
import matplotlib.pyplot as plt
import numpy as np

from id3 import ID3
from data import *

#d_sep_nov, n_sep_nov = load_data(DATA_2024,9,11)


'''
    We build a decision tree for each energy production type.
    After that, we take each day from december 2024 and input it into each decision tree.
    We sum all of the values, subtract the predicted energy used, that is our predicted sold.


'''

''' Now, each category differs in the time period that must be taken into account for:
    nuclear      : we take the previous year same time period.
    consum       : we take the previous year same time period.
    ape          : we take the previous year same time period.

    carbune      : we take this year and the previous year january and november

    foto         : we take the 3 previous years same time period
    eolian       : we take the 3 previous years same time period
    hidrocarburi : we take the 3 previous years same time period

    biomas       : we take the 2 previous months of the same year


'''

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
nuclear_t       = ID3(nuclear,7)
hidrocarburi_t  = ID3(hidrocarburi,7)
ape_t           = ID3(ape,30)

foto_t          = ID3(foto,3)


eolian_t        = ID3(eolian,14)

carbune_t       = ID3(carbune,30)
biomas_t        = ID3(biomas,7)

full_pred, full_sum = 0,0

#print(d.shape)
split_pred = np.zeros(11)
split_real = np.zeros(11)
for i in range(dtest.shape[0]):
    sum = 0
    date = dtest[i][2]

    #print()
    #print(date)

    '''
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


print(n)
res = np.array((split_real / split_pred) * 100, dtype='float')
n = n.flatten()
n = n[1:]
for label, percentage in zip(n,res):
    print(f'{label} => {percentage}')
print('\nPREDICTION',full_pred,full_sum, sep='\n')
#print(split_real)
#print(split_pred)
#print(n)

'''
print(np.array((split_real/split_pred)*100,dtype='float'))
'''




#remove date, it is not relevant?
#data = np.array(d_sep_nov[:,1:])
#d,l = split_label(data,10)
#tree = ID3(d,l,20)


print('Loaded data set')


'''
best results for december
lunile: ianuarie, februarie, decembrie | antreneare | anii 2021-2024
50 split treshold

best results for november
lunile: noiembrie, decembrie, ianuarie
50 split treshold

best result for july
lunie:  mai, iunie, iulie

'''


'''

dates, _        = extract_data_and_label(d,0,0)
_, consum       = extract_data_and_label(d,0,1)
_, carbune      = extract_data_and_label(d,0,4)
_, hidrocarburi = extract_data_and_label(d,0,5)
_, ape          = extract_data_and_label(d,0,6)
_, nuclear      = extract_data_and_label(d,0,7)
_, eolian       = extract_data_and_label(d,0,8)
_, foto         = extract_data_and_label(d,0,9)
_, biomas       = extract_data_and_label(d,0,10)


Test 1:
Randomly modifying time periods in order to match the targets
was a bad practice. I tried to match december 2023.

Test 2: Trying to tweak the parameters(ID3 train input) in order
to minimize overall error for every month of the year 2024


'''