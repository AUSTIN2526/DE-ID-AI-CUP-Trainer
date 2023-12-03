"""
AI CUP最終版本
移除了最終方案以外的所有程式碼
by AUSTIN2526
"""

from config import config
from utils.load_model_tokenizer import load_model_tokenizer
from utils.build_tool import same_seeds, use_accelerator
from utils.dataset import DatasetLoader, open_file
from utils.preprocess_utils import generate_windows, reduce_phi_null_data, process_sample, InputOutputDataset
from utils.trainer import get_optimizer_scheduler, Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    x, y, mask = zip(*batch)

    input_ids = pad_sequence(x, padding_value=tokenizer.pad_token_id, batch_first=True)
    labels = pad_sequence(y, padding_value=-100, batch_first=True)
    attention_mask = pad_sequence(mask, padding_value=0, batch_first=True)

    return {
        'input_ids': input_ids,  
        'labels': labels,
        'attention_mask': attention_mask    
    }

# 訓練參數
args = config()
same_seeds(args.seed)

# 固定使用Qwen
model, tokenizer = load_model_tokenizer("Qwen/Qwen-14B" if args.data_type !='Time' else 'Qwen/Qwen-7B')
model = use_accelerator(model)

# 讀取訓練資料
Loader = DatasetLoader(
    data_dir=args.train_file_dir,
    answer_path=args.answer_path
)

# 整理訓練資料集
DEID_dataset = Loader(
    data_type=args.data_type,
    mode='Train',
)
print(f'資料讀取完畢, 總共有: {len(DEID_dataset)} 訓練資料')

# 通過sliding Windows建立資料
x_windows, y_windows = generate_windows(
    dataset=DEID_dataset,
    window_size=args.windows_size,
    step=args.step
)

# 依照特定比例過濾無PHI的資料
print('準備刪除資料中...')
filter_x, filter_y = reduce_phi_null_data(
    x=x_windows, 
    y=y_windows,
    ratio=args.filter_ratio
)

# 轉換成Qwen訓練格式
inputs, att_mask, labels = process_sample(
    tokenizer=tokenizer,
    filter_x=filter_x,
    filter_y=filter_y,
    prompt=open_file(args.prompt_path)
)

# 切割資料(部分重疊)
x_train, x_valid, y_train, y_valid, mask_train, mask_valid = train_test_split(
    inputs, labels, att_mask,
    train_size=args.split_ratio, 
    random_state=args.seed, 
    shuffle=True
)

# 建立Pytorch Dataset
trainset = InputOutputDataset(x_train, y_train, mask_train)
validset = InputOutputDataset(x_valid, y_valid, mask_valid)


# 建立Pytorch DataLoader (注意padding)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)

# 檢查資料
data = next(iter(train_loader))
input_ids = data['input_ids'][0]
labels = data['labels'][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)
for i in range(len(input_ids)):
    print(f"|{tokens[i]}|{input_ids[i]}|{labels[i]}|")
print("Decoded Text:\n", tokenizer.batch_decode(data['input_ids'], skip_special_tokens=True)[0])

# 使用AdamW與warmup + 
optimizer, scheduler = get_optimizer_scheduler(
    model=model,
    epochs=args.epochs,
    data_len=len(train_loader),
    warmup_ratio=args.warmup_ratio
)

train_loader, valid_loader, optimizer, scheduler = use_accelerator(train_loader, valid_loader, optimizer, scheduler)

# 訓練 (使用NEFtune)
trainer = Trainer(
    model=model,
    epochs=args.epochs,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    file_name='Qwen-14B' if args.data_type !='Time' else 'Time-Qwen-7B'
)

# 開始訓練
trainer.train()
