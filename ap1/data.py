import numpy as np
import openpyxl as pyxl
import datetime

def load_data(file_location: set, months: set):
    '''
    Loads the months from the chosen set of excel files.

    Args:
        file_location : set(str) - paths to files containing excel data.
        months        : set(int) - months to be extracted from the previous excel data.

    Returns:
        tuple of data and names of attributes

    '''
    
    d = []
    for location in file_location:
        data = []

        wb = pyxl.open(location,read_only=True)
        ws = wb.active  

        #we search for the final month, and then for the start month
        final_month_row = -1

        
        prev_date = -1
        temp_sum = np.zeros(11)
        for i, row in enumerate(ws.iter_rows(min_row=2,min_col=1,max_col=12)):
            val = row[0].value
            if(val == 'Data'):
                continue

            # 30-11-2024 16:14:32
            if val == None: break
            date = datetime.datetime.strptime(val,"%d-%m-%Y %H:%M:%S")

            if date.month not in months: continue

            if date.month == 2 and date.day == 29: continue

            if prev_date == -1:
                prev_date = date
            if prev_date.day != date.day: # takes all the intervals in a day and leaves just the day
                
                temp_sum /= 6.0

                temp_sum = np.insert(temp_sum,0,f'{prev_date.timetuple().tm_yday}')
                temp_sum = np.insert(temp_sum,0,prev_date.month)
                temp_sum = np.insert(temp_sum,0,prev_date.year)
                data.append(temp_sum)

                prev_date = date
                temp_sum = np.zeros(11)
            else:
                temp_sum += np.array(list(map(lambda x: int(str(x.value).replace('*','')),row[1:])))

        if type(d) == list:
            d = np.zeros(np.array(data).shape)
        d += data   

    d /= file_location.__len__()

    a_names = []
    for names in ws.iter_rows(min_row=1,max_row=1,min_col=1,max_col=12):
        a_names.append(list(map(lambda x: str(x.value).replace('*',''),names)))
    
    #d = np.array(data)
    n = np.array(a_names)
    
    return d, n



def extract_data_and_label(data:np.array, data_index, label_index):
    '''
    Splits from a dataset with n attributes in a tuple of n-1 attributes and 1 label.

    Returns:
        (attributes, label)
    '''
    l = np.array(data[:,label_index])
    l = l.reshape(l.size,1)

    d = np.array(data[:,data_index])
    d = d.reshape(d.size,1)

    return d, l

DATA_FOLDER = 'C:\\Users\\Alex\\Desktop\\Facultate\\Anul 3\\ML\\ML_ap1\\ap1\\data'

DATA_2019 = DATA_FOLDER + '\\data2019.xlsx'
DATA_2020 = DATA_FOLDER + '\\data2020.xlsx'
DATA_2021 = DATA_FOLDER + '\\data2021.xlsx'
DATA_2022 = DATA_FOLDER + '\\data2022.xlsx'
DATA_2023 = DATA_FOLDER + '\\data2023.xlsx'
DATA_2024 = DATA_FOLDER + '\\data2024.xlsx'
DATA_2024_DEC = DATA_FOLDER + '\\data2024_dec.xlsx'
DATA_2021_2024 = DATA_FOLDER + '\\data2021_2024.xlsx'

DATA_SETS = [
    DATA_2019,
    DATA_2020,
    DATA_2021,
    DATA_2022,
    DATA_2023,
    DATA_2024,
    DATA_2024_DEC
    ]
