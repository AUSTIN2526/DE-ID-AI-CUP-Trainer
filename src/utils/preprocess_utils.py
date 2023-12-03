import numpy as np
from torch.utils.data import Dataset
import torch

def generate_windows(dataset, window_size=2, step=1):
    x_windows, y_windows = [], []

    for data in dataset.values():
        for i in range(0, len(data), step):
            context, label = [], []

            start_index = max(0, i - window_size)
            end_index = min(i + window_size + 1, len(data))
            window_indices = range(start_index, end_index)

            # Build context and labels
            for j in window_indices:
                context.append(data[j][0])
                if data[j][1] != 'PHI|NULL':
                    label.append(data[j][1])

            # 當label為空時，設定成PHI|NULL
            y_windows.append("\n".join(label) or 'PHI|NULL')

            x_windows.append("\n".join(context))

    return x_windows, y_windows


def reduce_phi_null_data(x, y, ratio=0.7):
    x, y = np.array(x), np.array(y)

    NULL_IDX = (y == 'PHI|NULL') 

   
    NULL_x = x[NULL_IDX]
    NULL_y = y[NULL_IDX]
    PHI_x = x[~NULL_IDX]
    PHI_y = y[~NULL_IDX]
    print(f'有PHI的訓練資料: {len(PHI_x)}, 沒有PHI的訓練資料: {len(NULL_x)}')


    random_indices = np.random.choice(len(NULL_x), int(len(PHI_x) * ratio), replace=False)
    selected_NULL_x = NULL_x[random_indices]
    selected_NULL_y = NULL_y[random_indices]

    # 合併資料
    filter_x = np.concatenate((PHI_x, selected_NULL_x), axis=0)
    filter_y = np.concatenate((PHI_y, selected_NULL_y), axis=0)
    print(f'資料過濾完畢剩餘: {len(filter_x)}')

    return filter_x, filter_y

def sft_format(tokenizer, inputs, output=None, prompt=''):
    prompt = f'### PROMPT\n{prompt}\n\n### INPUT\n{inputs}\n\n### OUTPUT\n' 

    # 輸入的部分bos token可有可無，給予Tokenizer自行處理
    input_ids = tokenizer.encode(
        text=prompt, 
        add_special_tokens=True,
        truncation=True,
        max_length=8192,
        return_tensors='pt'
    )

    if output is not None:
        # 輸出則不能有任何特殊標記，因為eos我手動加入
        outputs_ids = tokenizer.encode(
            text=output, 
            add_special_tokens=False, 
            truncation=True,
            max_length=8192,
            return_tensors='pt'
        )

        # 手動加入結尾
        eos = torch.tensor([[tokenizer.eos_token_id]])

        # 模型的標籤 (需mask掉input_ids)
        label_mask = torch.full((input_ids.shape[0], input_ids.shape[-1]), -100) 
        labels = torch.cat((label_mask, outputs_ids, eos), dim=-1)

        # 模型輸入
        input_ids = torch.cat((input_ids, outputs_ids, eos), dim=-1)

        # 模型的attention_mask
        attention_mask = torch.full((input_ids.shape[0], input_ids.shape[-1]), 1)

        
        
        # 檢查context windows是否超出最佳Attention的長度
        if input_ids.shape[-1] > 8192:
            raise ValueError(f'i輸入的Token大小: {inputs.shape[-1]} 超出了最佳Attention的長度')
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    # 若沒輸入Output attention_mask的大小只會為input_ids
    attention_mask = torch.full((input_ids.shape[0], input_ids.shape[-1]), 1)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def process_sample(tokenizer, filter_x, filter_y, prompt=''):
    # 快速整理格式用
    result_list = []
    for x, y in zip(filter_x, filter_y):
        inputs, labels, att_mask = sft_format(tokenizer, x, y, prompt).values() 
        result_list.append([inputs[0], labels[0], att_mask[0]])

    return zip(*result_list)


class InputOutputDataset(Dataset):
    def __init__(self, x, y, mask):
        self.x = x
        self.y = y
        self.mask = mask

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]
       
    def __len__(self):
        return len(self.x)



