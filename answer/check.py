import os
import re

def validate_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        lines = [i.strip().split('\t') for i in file.readlines()]

    for line in lines:
        try:
            file_name, PHI, start_pos, end_pos, actual_label = line
        except:
            file_name, PHI, start_pos, end_pos, actual_label, time = line
        # 使用正規表達式提取資訊
  
        # 使用相對路徑構建測試資料檔案的路徑
        test_data_path = os.path.join('../dataset/test_data', file_name + '.txt')

        # 讀取測試資料檔案
        with open(test_data_path, 'r', encoding='utf-8-sig') as test_file:
            content = test_file.read()

        # 驗證起始位子和結束位子是否一致
        #
        if content[int(start_pos):int(end_pos)] != actual_label:
            print(f"檔案 {file_name} 中的起始位子和結束位子與實際標籤不一致。")
            print(content[int(start_pos):int(end_pos)], actual_label)

# 指定檔案路徑
file_path = 'answer.txt'

# 呼叫函式進行驗證
validate_file(file_path)
