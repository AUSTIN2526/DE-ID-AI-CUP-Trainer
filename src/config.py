import argparse

def config():
    parser = argparse.ArgumentParser(description="這裡只是方便我切換")
    
    parser.add_argument("--adapter_name",
                        help="推理時需要使用的adapter名稱",
                        default='Qwen-14B_4')
    
    parser.add_argument("--train_file_dir",
                        help="訓練集資料夾",
                        default='./dataset/train_data')
    
    parser.add_argument("--valid_file_dir",
                        help="驗證及資料夾",
                        default='./dataset/valid_data')
    
    parser.add_argument("--answer_path",
                        help="答案的文件",
                        default='./answer/train_answer.txt')
    
    parser.add_argument("--prompt_path",
                        help="prompt的文件",
                        default='./prompt/De-ID.txt')
    
    parser.add_argument("--out_dir",
                        help="答案輸出位子",
                        default='answer/answer.txt')
    
    parser.add_argument("--data_type",
                        help="訓練模式，只有De-ID(去識別化)，跟Time(時間正規化)",
                        default='De-ID')
                        
    parser.add_argument("--seed",
                        help="固定亂數",
                        type=int,
                        default=2526)

    parser.add_argument("--windows_size",
                        type=int,
                        default=1,
                        help="通過sliding結合上下文")

    parser.add_argument("--step",
                        type=int,
                        default=1,
                        help="Windows移動步長")

    parser.add_argument("--epochs",
                        type=int,
                        default=5,
                        help="訓練總數")

    parser.add_argument("--filter_ratio",
                        type=float,
                        default=0.1,
                        help="無效PHI過濾比率")

    parser.add_argument("--split_ratio",
                        type=float,
                        default=0.8,
                        help="訓練與驗證過濾比例")

    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="每次給予模型的資料量")

    parser.add_argument("--warmup_ratio",
                        type=float,
                        default=0.2,
                        help="第一個Epoch中Warmup的比例")

    args = parser.parse_args()

    return args
