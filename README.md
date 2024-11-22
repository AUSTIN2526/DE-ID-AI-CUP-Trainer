# DE-ID-AI-CUP-Trainer
2023 AI CUP **Privacy protection and medical data standardization competition: decoding clinical cases and letting data tell stories** dedicated code

## Description
We utilized the Qwen-14B and 7B models for the subtasks 1 and 2, respectively, based on the findings by Wang, Bi and Zhu, Ren, which indicated that Qwen models perform well in handling clinical notes. To enhance training stability for these large models, we implemented the fully sharded data parallel technique. To mitigate potential overfitting, we employ the NEFTune method, which introduces uniformly distributed noise to the model’s embedding layer. This noise helps prevent the model from memorizing specific details within the training set, allowing it to generalize more effectively to new data. Additionally, this technique reduces the model’s sensitivity to particular inputs, preventing it from developing overly complex representations too early in the training process. Another strategy we employed was randomly extracting sentences within a predefined context window and utilizing them as a basis for validating the model’s performance. The length of the context window was set to 1 for subtask 1, but 3 for subtask 2, as the normalization of temporal information often requires more contextual data. During the inference phase, we used a greedy decoding strategy to ensure the stability and reliability of the model’s output. In addition, we incorporated a rule-based post-processing method since we observed that the fine-tuned model occasionally makes erroneous predictions for some labels that should be straightforward to classify.

![https://github.com/AUSTIN2526/DE-ID-AI-CUP-Trainer/blob/main/speech.jpg](https://github.com/AUSTIN2526/DE-ID-AI-CUP-Trainer/blob/main/speech.jpg)


## Competition Ranking
* **Private Leaderboard Rank:** 2
* **Task 1 Score:** 0.9075701 
* **Task 2 Score:** 0.8124762 

### Note
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

## Certificate of Merit
![https://github.com/AUSTIN2526/DE-ID-AI-CUP-Trainer/blob/main/award.jpg](https://github.com/AUSTIN2526/DE-ID-AI-CUP-Trainer/blob/main/award.jpg)



## Team Members & Announcement
* Team Members (I'm Team Leader)
![https://github.com/AUSTIN2526/DE-ID-AI-CUP-Trainer/blob/main/team_3951.png](https://github.com/AUSTIN2526/DE-ID-AI-CUP-Trainer/blob/main/team_3951.png)
Announcement: [Link](https://www.aicup.tw/post/%E3%80%90%E5%BE%97%E7%8D%8E%E5%90%8D%E5%96%AE%E3%80%91ai-cup-2023-%E7%A7%8B%E5%AD%A3%E8%B3%BD-%E3%80%8C%E9%9A%B1%E7%A7%81%E4%BF%9D%E8%AD%B7%E8%88%87%E9%86%AB%E5%AD%B8%E6%95%B8%E6%93%9A%E6%A8%99%E6%BA%96%E5%8C%96%E7%AB%B6%E8%B3%BD%EF%BC%9A%E8%A7%A3%E7%A2%BC%E8%87%A8%E5%BA%8A%E7%97%85%E4%BE%8B%E3%80%81%E8%AE%93%E6%95%B8%E6%93%9A%E8%AA%AA%E6%95%85%E4%BA%8B%E3%80%8D)

## 這比賽助理哪來的臉說他是我學長
我嚴重懷疑這次比賽平台的助理的公關能力與個人能力。對於回報最終排名計算方式出錯的問題，我總共等了28天才收到一句「正在確認中」的回覆，但收到這個回覆時，比賽成績早就準備公布了，我還需要這句話嗎？參與過AI CUP的人應該都知道，這個比賽是分階段進行的，但各階段公布時間常常會延後30分鐘到1小時，時間彷彿只是參考用的。且在公布排名時，直接將所有參賽者的郵件地址按照排名順序批量CC發送，這樣不是會導致個人資料外洩嗎？這次比賽的主題甚至就是防範個資外洩啊！提供的程式碼也不知是哪裡抄來的，充滿了繁瑣且無用的區段，維護性幾乎為零程式寫成這樣就不要寫了。  

這次比賽的資料集雖有許多錯誤，而比賽助理的用處就是在最短的時間內解決參賽方所提出的問題。在比賽的過程中我總共修正了30個以上的資料集問題。而且在我回報問題時一個問題需要花費一個星期以上才收到回覆在整個過程中，我在比賽平台提出了4次問題，電子郵件聯絡了5次，總共9次。這些問題主要針對資料集錯誤、計算方法有誤及人員顯示錯誤等方面進行提問。然而這些等待回復的時間加起來共花費了超過40天，且在郵件聯絡中我只獲得了一次回復。  

我甚至懷疑這個助理是否真的具備程式設計的能力以及在這一領域的專業知識。在查看他的碩士畢業論文後，我才確定他確實沒有這項能力。使用LLaMA-Factory來進行訓練，效果都會比較好。我不清楚你寫這些程式到底有什麼用意，如果連基本知識都不懂，那乾脆不要學AI了。怎麼會有人的訓練結果是12B的模型效果會遠低於7B？稍微有點常識的人都會猜測可能是因為超參數設置導致梯度爆炸然後去修正問題吧?我甚至在同個模型下用70M的模型比你7B和12B高出2%、3%的成績。使用BERT甚至也能超過10%的效果。  

**我很感謝本次比賽的相關機構與導師，讓我有這次的機會能夠參與並獲得獎項，但請別讓一位助理毀了整個比賽項目**
