import os
import numpy as np

from id3 import ID3
from data import *


def section(e,d):
    return np.array(list(filter(e,d)))

months = {1,2,3,4,5,6,7,8,9,10,11,12}
d2024, n = load_data([DATA_2024],months)
d2024_DEC, _ = load_data([DATA_2024_DEC],{12})
d2021_2022_2023, _ = load_data([DATA_2021,DATA_2022,DATA_2023],months)


dtest = section(lambda x: x[1] in {12}, d2024_DEC)

#sold
ds = extract_data_and_label(section(lambda x: x[1] in months, d2021_2022_2023),2,13)
ids = ID3(ds,3)

total = 0

total_real = 0
total_pred = 0

for i in range(dtest.shape[0]):
    date = dtest[i][2]

    pred = ids.predict([date])
    real = dtest[i][13]

    total_real += real
    total_pred += pred

    total += (pred-real)**2
    pass

total /= dtest.shape[0]
total = np.sqrt(total)

#print(total) RMSE
print('PREDICTION')
print(total_pred)
print(total_real)
'''
idco = ID3(dco,7)
idn  = ID3(dn, 7)
idh  = ID3(dh,7)
ida  = ID3(da,7)
idf  = ID3(df,7)
idca = ID3(dca,7)
ido  = ID3(do,7)
idb  = ID3(db,7)
'''

