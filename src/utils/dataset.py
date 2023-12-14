from pathlib import Path
from collections import defaultdict

def open_file(path, readlines=False):
    with open(path, 'r', encoding='utf-8-sig') as file:
        if readlines:
            return file.readlines()
        return file.read()


class DatasetLoader:
    def __init__(self, data_dir=None, answer_path=None):
        self.data_dir = data_dir
        self.answer_path = answer_path

    def __call__(self, data_type, mode='Train'):
        # 去識別化模式
        if data_type == 'De-ID' and mode in ['Train', 'Valid']:
            if self.data_dir:  # 檢查輸入路徑
                if mode == 'Train' and not self.answer_path:
                    raise ValueError(f'Data type: {data_type} Mode: {mode} requires "answer_path"')
                return self.processing_de_id_data(mode)
            else:
                raise ValueError(f'Data type: {data_type} Mode: {mode} requires "data_dir"')
        
        # 時間正規化模式
        elif data_type == 'Time' and mode in ['Train', 'Valid']:
            if self.data_dir:
                if mode == 'Train' and not self.answer_path:
                    raise ValueError(f'Data type: {data_type} Mode: {mode} requires "data_dir"')
                else:
                    return self.processing_time_data(mode)
                
        # 輸入錯誤
        else:
            raise ValueError(f'Invalid data_type: {data_type} or mode: {mode}')

    def processing_time_data(self, mode):
        path = Path(self.data_dir)
        train_data = {file.stem: open_file(file) for file in path.iterdir() if file.is_file()}

        # 驗證集用於推理，因此直接返回資料
        if mode == 'Valid':
            return train_data

        # 整理成dict並過濾資料
        train_answer = defaultdict(list)
        for line in open_file(self.answer_path, True):
            split_line = line.strip().split('\t')
            head, body = split_line[0], split_line[1:]
            if body[0] in ['DATE', 'TIME', 'DURATION', 'SET']:
                train_answer[head].append(body)

        # 整理答案
        return self.integrate_data(train_data, train_answer, mode)

    def processing_de_id_data(self, mode):
        # 理論上要跟上面的合併，這部份是我一直修改程式後留下來的垃圾區段
        path = Path(self.data_dir)
        train_data = {file.stem: open_file(file) for file in path.iterdir() if file.is_file()}
        
        # 驗證集用於推理，因此直接返回資料
        if mode == 'Valid':
            return train_data

        # 整理成dict並過濾資料
        train_answer = defaultdict(list)
        for line in open_file(self.answer_path, True):
            split_line = line.strip().split('\t')
            head, body = split_line[0], split_line[1:]
           
            if body[0] not in ['DATE', 'TIME', 'DURATION', 'SET']:
                if len(body) > 4:
                    body = body[:3] + ["\t".join(body[3:])] # 有部分資料集通過\t分開
                train_answer[head].append(body)

        return self.integrate_data(train_data, train_answer)
    

    def integrate_data(self, file, answers, mode='De-ID'):
        integrated_dataset = defaultdict(list)

        for key, values in file.items():
            lines = values.split('\n')
            line_offset, answer_index = 0, 0

            for line in lines:
                labels = []
                if answer_index < len(answers[key]):
                    
                    start, end = map(int, answers[key][answer_index][1:3])
                    
                    while answer_index < len(answers[key]):
                        start, end = map(int, answers[key][answer_index][1:3])


                        if line_offset + len(line) + 1 < end:
                            break

                        
                        label = answers[key][answer_index][0]
                        label_text = line[start - line_offset:end - line_offset]

                        # 防呆
                        if label_text != answers[key][answer_index][3]:
                            #print(start - line_offset)
                            #print()
                            raise ValueError(f'資料集錯誤,在資料{key}中, 程式找到了 {label_text} 但答案應該為 {answers[key][answer_index][3]}')
                           
                        if mode == 'De-ID':
                            labels.append(f'{label}|{label_text}')
                        else:
                            labels.append(f'{label}|{label_text}|{answers[key][answer_index][-1]}')
                        answer_index += 1

                # 移除空白
                if line.strip() != '':
                    if mode == 'De-ID':
                        integrated_dataset[key].append([line, "\n".join(labels) if labels != [] else 'PHI|NULL'])
                    else:
                        integrated_dataset[key].append([line, "\n".join(labels) if labels != [] else 'PHI|NULL'])

                # 更新文件起始位子
                line_offset += len(line) + 1

        return integrated_dataset

            





    
