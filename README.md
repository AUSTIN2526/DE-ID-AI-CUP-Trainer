# DE-ID-AI-CUP-Trainer
2023年AI CUP**隱私保護與醫學數據標準化競賽：解碼臨床病例、讓數據說故事**專用程式碼
## 注意
注意該資料集因DUA協議，無法上傳至公開網站中，請自行替換以下路徑中的檔案
```
answer/train_ansewer.txt
answer/valid_ansewer.txt
dataset/train_data
dataset/valid_data
dataset/test_data
```

有修正的訓練與驗證資料集
```
1481, 1139, 1059, 661, 830, 943 資料錯誤或不存在
1071, 111, 583(多處), 834(多處) 索引值錯誤
1906, 377, file20783 字元混亂
```

## 環境
* 作業系統：Windows 11
* 程式語言：Python 3.8.10
* CPU規格：Intel(R) Core(TM) i9-10900 CPU 2.80GHz
* GPU規格：ASUS TURBO RTX 3090 TURBO-RTX3090-24G
* CUDA版本：12.2

## 安裝函式庫
* [CUDA 11.6](https://www.nvidia.com/zh-tw/geforce/technologies/cuda/) 或以上版本
* [PyTorch 1.12](https://pytorch.org/) 或以上版本
* [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) (選用)

以上環境安裝完畢後執行
```
pip install -r requirements
```

## 使用說明
該程式主要分為以下三個部分第一部分為訓練程式碼、第二部分為推理程式碼、第三階段則為過濾器程式碼
1. ### 訓練參數設定
* 訓練去識別化
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
* 訓練時間正規化
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
2. ### 推理參數設定
* 推理去識別化
```
python src/predict.py ^
--adapter_name "Qwen-14B_4" ^
--prompt_path "./prompt/De-ID.txt" ^
--data_type "De-ID" ^
--windows_size 1
```
* 推理時間正規化
```
python src/predict.py ^
--adapter_name "Time-Qwen-7B_4" ^
--prompt_path "./prompt/Time.txt" ^
--data_type "Time" ^
--windows_size 3
```
3. ### 過濾時間正規化資料
* 執行以下兩個步驟 (此步驟請先再推理去識別化前完成)
```
cd answer
py filter.py
```
