# DE-ID-AI-CUP-Trainer
2023 AI CUP **隱私保護與醫學數據標準化競賽：解碼臨床病例、讓數據說故事** dedicated code

## Competition Ranking
* **Private Leaderboard Rank:** 2
* **Task 1 Score:** 0.9075701 
* **Task 2 Score:** 0.8124762 

## Note
Due to the DUA agreement, the dataset cannot be uploaded to public websites. Please replace the files in the following paths on your own:
```
answer/train_ansewer.txt
answer/valid_ansewer.txt
dataset/train_data
dataset/valid_data
dataset/test_data
```

Modified training and validation datasets:
```
1481, 1139, 1059, 661, 830, 943 Data errors or do not exist
1071, 111, 583 (multiple), 834 (multiple) Index errors
1906, 377, file20783 Character confusion
```

## Data Format
The format for the training and validation `answer.txt` is as follows:
```
file_name    PHI    start_idx   end_idx   target_text (separated by tabs)
                        .
                        .
file_name    PHI    start_idx   end_idx   target_text
```

The content of the `dataset` folder is structured as:
```
  |-train_data
  |    |-9.txt
  |    |-10.txt
  |       .
  |       .
  |    |-file233771.txt
  |
  |-valid_data
  |     |-24.txt
  |     |-47.txt
  |        .
  |        .
  |     |-file30810.txt
```

## Environment
* Operating System: Windows 11
* Programming Language: Python 3.8.10
* CPU Specifications: Intel(R) Core(TM) i9-10900 CPU 2.80GHz
* GPU Specifications: ASUS TURBO RTX 3090 TURBO-RTX3090-24G
* CUDA Version: 12.2

## Library Installation
* [CUDA 11.6](https://www.nvidia.com/zh-tw/geforce/technologies/cuda/) or later
* [PyTorch 1.12](https://pytorch.org/) or later
* [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) (optional)

After installing the above environment, execute the following command:
```
pip install -r requirements
```

## Usage Instructions
The program is mainly divided into three parts: training code, inference code, and filter code.

1. ### Training Parameter Configuration
* Train de-identification:
```
python src/train.py ^
--prompt_path "./prompt/De-ID.txt" ^
--data_type "De-ID" ^
--windows_size 1 ^
--batch_size 4 ^
--seed 2526 ^
--epochs 10 ^
--filter_ratio 0.1 ^
--split_ratio 0.8 ^
--warmup_ratio 0.2
```
* Train time normalization:
```
python src/train.py ^
--prompt_path "./prompt/Time.txt" ^
--data_type "Time" ^
--windows_size 3 ^
--batch_size 2 ^
--seed 2526 ^
--epochs 10 ^
--filter_ratio 0.1 ^
--split_ratio 0.8 ^
--warmup_ratio 0.2
```

2. ### Inference Parameter Configuration
* Inference de-identification:
```
python src/predict.py ^
--adapter_name "Qwen-14B_9" ^
--prompt_path "./prompt/De-ID.txt" ^
--data_type "De-ID" ^
--windows_size 1
```
* Inference time normalization:
```
python src/predict.py ^
--adapter_name "Time-Qwen-7B_9" ^
--prompt_path "./prompt/Time.txt" ^
--data_type "Time" ^
--windows_size 3
```

3. ### Filter Time-Normalized Data
* Execute the following two steps (complete this step before inference de-identification):
```
cd answer
py filter.py
```
