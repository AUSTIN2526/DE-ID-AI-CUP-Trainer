import re
from collections import defaultdict

# 讀取txt檔案
with open('answer.txt', 'r', encoding='utf-8-sig') as file:
    lines = file.readlines()

# 定義資料結構儲存結果
data_dict = defaultdict(list)

# 解析每一行資料
for line in lines:
    parts = line.strip().split('\t')
    filename, label, start, end, data, normalized_result = parts[0], parts[1], int(parts[2]), int(parts[3]), parts[4], parts[5]

    # 儲存結果到字典
    data_dict[filename].append((label, start, end, data, normalized_result))

# 合併與計算結果
for filename, results in data_dict.items():
    merged_results = []
    for result in results:
        label, start, end, data, normalized_result = result

        # 檢查是否需要使用規則三
        use_rule_3 = label in ['TIME', 'DATE']

        # 檢查是否有重疊
        overlap = False
        for merged_result in merged_results:
            _, merged_start, merged_end, _, _ = merged_result
            if start <= merged_end and end >= merged_start:
                overlap = True
                # 開頭或結尾其中一項重疊，視情況保留字串最長的結果
                if use_rule_3 and len(normalized_result) > len(merged_result[4]):
                    merged_results.remove(merged_result)
                    merged_results.append(result)
                elif not use_rule_3:
                    # 不需要使用規則三的情況下，直接合併
                    merged_results.remove(merged_result)
                    merged_results.append(result)
                break
        
        # 如果沒有重疊，則直接加入合併結果
        if not overlap:
            merged_results.append(result)

    # 輸出結果
    with open('answer.txt', 'w', encoding='utf-8') as output_file:
        for merged_result in merged_results:
            output_file.write(f"{filename}\t{merged_result[0]}\t{merged_result[1]}\t{merged_result[2]}\t{merged_result[3]}\t{merged_result[4]}\n")