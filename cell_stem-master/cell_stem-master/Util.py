

import numpy as np
import os
import gzip, pickle




# ------------------------------------------------------------------------
# make directory
def mkdir(path):
    try:
        os.makedirs(path)
    except Exception as ex:
        print(str(ex))



# ------------------------------------------------------------------------
# save data (compressed)
def save_data(obj, file_name):
    file = gzip.GzipFile(file_name, 'wb')
    pickle.dump(obj, file)
    return


def load_data(file_name):
    file = gzip.GzipFile(file_name, 'rb')
    obj = pickle.load(file)
    return obj



