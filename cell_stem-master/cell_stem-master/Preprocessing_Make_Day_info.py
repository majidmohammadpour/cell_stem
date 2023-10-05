

import os
import pandas as pd



'''
Train Data의 Day 정보를 one-hot 형태로 저장
 - Day 숫자 1, 2, 3... 에 따라 비례하여 영향을 주는 것이 아니기 때문에 one-hot 형태로 적용 필요 
 - 0.5일 단위 정보 저장
'''

DATA_TYPE = 'prep_data2'                            # Threshold + Histogram Equlize 적용한 Trainset
DATA_PATH = '../dataset/{}/'.format(DATA_TYPE)      #
SAVE_PATH = DATA_PATH                               # Day Info를 저장할 디렉토리

# 날짜별 one-hot array의 index지정 (0~19)
DAYS_DIC = {
    'day0': 0, 'day0.5': 1, 'day1': 2, 'day1.5': 3, 'day2': 4,
    'day2.5': 5, 'day3': 6, 'day3.5': 7, 'day4': 8, 'day4.5': 9,
    'day5': 10, 'day5.5': 11, 'day6': 12, 'day6.5': 13, 'day7': 14,
    'day7.5': 15, 'day8': 16, 'day8.5': 17, 'day9': 18, 'day9.5': 19
}


# =======================================================================================
# Train Data 디렉토리와 파일명 Parsing하여 Class와 Day 정보 저장


days_info_list = []
for dirname, subdirs, files in os.walk(DATA_PATH):
    if 'train' not in dirname:                      # train data에 한함
        continue
    # -------------------------------------------------------
    dirname = dirname.replace('\\', '/').replace(' ', '_')
    dir_splits = dirname.split('/')
    cell_class = dir_splits[-1]         # class type
    for filename in files:
        if '00000.jpg' not in filename:
            continue
        # ---------------------------------------------------
        file_splits = filename.split('_')
        week = file_splits[0].lower()
        plate = file_splits[1].lower()
        col_num = file_splits[-2][1:].lower()
        # ---------------------------------------------------
        # 예외처리: 날짜가 지나도 class1 상태 변하지 않은 cell 제외
        if week == '1' and plate == 'plate1' and col_num in ['07', '08', '09', '10', '11', '12']:
            print('Remove:', filename)
            continue
        # ---------------------------------------------------
        day_one_hot = [0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0]
        day = file_splits[-3].strip().lower()
        day_index = DAYS_DIC.get(day)
        if day_index is None:
            continue
        day_one_hot[day_index] = 1
        info = [dirname + '/', filename, cell_class, day] + day_one_hot
        days_info_list.append(info)




# =======================================================================================
# Day Info를 csv 파일로 저장
pd_days_info_list = pd.DataFrame(days_info_list, index=None)
pd_days_info_list.columns = [
    'dirname', 'filename', 'class', 'day',
    'day0', 'day0.5', 'day1', 'day1.5', 'day2', 'day2.5', 'day3', 'day3.5', 'day4', 'day4.5',
    'day5', 'day5.5', 'day6', 'day6.5', 'day7', 'day7.5', 'day8', 'day8.5', 'day9', 'day9.5'
]
pd_days_info_list.to_csv(SAVE_PATH + 'Days_Info.csv', index=None)



