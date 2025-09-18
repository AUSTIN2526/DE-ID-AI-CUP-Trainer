import torch

class TextInference:
    def __init__(self, model, tokenizer, output_directory, window_limit, prompt, sft_formatter, data_type):
        self.model = model
        self.tokenizer = tokenizer
        self.output_directory = output_directory
        self.window_size = (window_limit * 2) + 1
        self.sft_formatter = sft_formatter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt = prompt
        self.data_type = data_type
        self.labels = ['ZIP', 'CITY', 'COUNTRY', 'STATE', 'DATE', 'PATIENT', 'DEPARTMENT', 'DURATION', 'IDNUM', 'AGE', 'PHONE', 'MEDICALRECORD', 'SET', 'ROOM', 'URL', 'ORGANIZATION', 'TIME', 'STREET', 'HOSPITAL', 'DOCTOR', 'LOCATION-OTHER']
 
    def __call__(self, test_file, sliding = True):
        if sliding:
            for file_name, content in test_file.items():
                output_answers = []                    # 暫存每份文件的答案
                content_inputs, model_inputs = [], []  # 模型實際輸入與實際文本
                line_offset, windows_num = 0, 0        # 每次迭代初始化參數

                # 根據換行福切割文本
                for text in content.split('\n'):
                    content_inputs.append(text)   # 紀錄所有文本資料
                    if text.strip() != '':
                        model_inputs.append(text) # 只紀錄模型輸入
                        windows_num += 1
                    
                    
                    if self.window_size == windows_num:
                        # 計算答案並更新文本偏移量
                        answer = self.extract_answer(file_name, line_offset, content_inputs, model_inputs)
                        line_offset += len("\n".join(content_inputs)) + 1

                        # 確保有答案
                        if answer is not None:
                            output_answers.extend(answer)
                        
                        # 重製
                        content_inputs, model_inputs = [], []
                        windows_num = 0

                # 不足的條件在計算一次
                if self.window_size != windows_num:
                    answer = self.extract_answer(file_name, line_offset, content_inputs, model_inputs)
                    line_offset += len("\n".join(content_inputs)) + 1
                    if answer is not None:
                        output_answers.extend(answer)

                # 存檔
                if output_answers!=[]:
                    self.save_output(self.output_directory, output_answers)
        else:
            for file_name, content in test_file.items():
                output_answers = []  # Store the output answers for each file
                content_inputs, model_inputs = [], []  # Lists to store content and model inputs
                line_offset = 0  # Initialize line_offset

                # Split the content into lines
                lines = content.split('\n')

                # Iterate through each line in the content using a sliding window
                for i in range(len(lines) - self.window_size + 1):
                    window_lines = lines[i:i+self.window_size]
                    
                    # Process each line in the window
                    for text in window_lines:
                        content_inputs.append(text)
                        if text.strip() != '':
                            model_inputs.append(text)
                        
                    # Extract answers for the current window
                    answer = self.extract_answer(file_name, line_offset, content_inputs, model_inputs)
                    line_offset += len(content_inputs[0]) + 1
                    if answer is not None:
                        output_answers.extend(answer)

                    # Reset the inputs for the next window
                    content_inputs, model_inputs = [], []

                # Save the output answers to the specified directory
                if output_answers != []:
                    self.save_output(self.output_directory, output_answers)

    def extract_answer(self, file_name, line_offset, content_inputs, model_inputs):
        content_answers, error_answers = [], []
        #idx_buffer = []
        
        model_inputs_str = "\n".join(model_inputs)
        content_inputs_str = "\n".join(content_inputs)

        # 模型預測
        predictions_and_text = self.predict(model_inputs_str)
        last_ans_position = 0

        # 迭代每個答案
        for prediction_and_target in predictions_and_text:

            # 去識別化
            if self.data_type == 'De-ID':
                prediction, target = self.parse_prediction_and_target(prediction_and_target)
            
            # 時間正規化
            else:
                prediction, target, norm_time = self.parse_prediction_and_target(prediction_and_target)
            
            # 忽略生成錯誤的結果與PHI|NULL
            if target != 'NULL' and prediction is not None:

                # 計算每個答案的位子並整理格式
                if self.data_type == 'De-ID':
                    start, last_ans_position, answer = self.calculate_answer_offset(file_name, line_offset, last_ans_position, content_inputs_str, prediction, target)
                else:
                    start, last_ans_position, answer = self.calculate_answer_time_offset(file_name, line_offset, last_ans_position, content_inputs_str, prediction, target, norm_time)
                
                # 找到答案
                if answer is not None:
                    #idx_buffer.append([line_offset+start, line_offset+last_ans_position])
                    content_answers.append(answer)

                # 找不到則暫存這些資訊
                else:
                    error_answers.append([file_name, line_offset, last_ans_position, content_inputs_str, prediction, target])

        # 修正找不到的答案
        # corrected_answers = self.correct_predicted_names(error_answers, idx_buffer)
        # content_answers.extend(corrected_answers)
        
        return content_answers if content_answers else None

    def parse_prediction_and_target(self, prediction_and_target):
        # 整理資料
        try:
             # Split the combined string into parts using '|'
            if self.data_type == 'De-ID':
                parts = prediction_and_target.split('|')
                
                # Extract prediction and target, and strip leading/trailing whitespaces
                prediction = parts[0].strip()
                target = parts[1].strip()

                return prediction, target
            else:
                parts = prediction_and_target.split('|')
                prediction = parts[0].strip()
                time = parts[1].strip()
                norm_time = parts[2].strip()

                return prediction, time, norm_time
        
        except:
            if self.data_type == 'De-ID':
                return None, None
            else:
                return None, None, None
        
    def calculate_answer_offset(self, file_name, line_offset, last_ans_position, content_inputs, prediction, target):
        # 目標文字
        target = target.strip()

        # 找尋目標文字的起始值
        start = content_inputs.find(target, last_ans_position)
        end = start + len(target)

        # 若有被找到且序列符合
        if start != -1 and content_inputs[start:end] == target and target and len(target) > 1 and prediction in self.labels:
            # 轉換答案格式

            corr = ''
            if prediction =='IDNUM' and target.find('.') != -1:
                prediction = 'MEDICALRECORD'
                corr ='(修正)'

            ## 自行根據模型加入過濾守則
                
            answer_format = f'{file_name}\t{prediction}\t{line_offset + start}\t{line_offset + end}\t{target}'
            print(answer_format, corr)
            return start, end, answer_format
        else:
            print('找不到', target)
            return start, last_ans_position, None
        
    def calculate_answer_time_offset(self, file_name, line_offset, last_ans_position, content_inputs, prediction, target, norm_time):
        # 理應跟上面合併，但我還是懶得改
        target = target.strip()
        start = content_inputs.find(target, last_ans_position)
        end = start + len(target)

        if start != -1 and content_inputs[start:end] == target and target.strip() != '' and len(target) !=1 and prediction in self.labels:
            answer_format = f'{file_name}\t{prediction}\t{line_offset + start}\t{line_offset + end}\t{target}\t{norm_time}'
            print(answer_format)
            return start, end, answer_format
        else:
            print('找不到', target)
            return start, last_ans_position, None

        
    def predict(self, model_inputs,): 
        # 轉換並生成文字(注意不要傳入output參數)
        formatted_inputs = self.sft_formatter(self.tokenizer, model_inputs, prompt=self.prompt)
        generated_text = self.generate(formatted_inputs)
        predictions_and_text = generated_text.split('### OUTPUT\n')[1].split('\n')

        return predictions_and_text

    def generate(self, formatted_inputs):
        # 生成文字
        with torch.inference_mode():
            formatted_inputs = {k: v.to(self.device) for k, v in formatted_inputs.items()}
            generated_sequence = self.model.generate(
                **formatted_inputs, 
                max_new_tokens=200,
                num_return_sequences=1, 
                do_sample=False, 
                num_beams=1,  # 設置大於1的值
                early_stopping=False,  # 取消 early_stopping
                pad_token_id=self.tokenizer.pad_token_id
            )


        generated_text = self.tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)[0]
        #print(generated_text)
        return generated_text
    
    def save_output(self, path, output_ans):
        with open(path, 'a', encoding='utf-8-sig') as f:
            if f.tell() > 0:  
                f.write('\n' + "\n".join(output_ans))
            else:
                f.write("\n".join(output_ans))


        
    """
    修正答案(視情況開啟)
    def correct_predicted_names(self, error_answers, idx_buffer):
        corrected_answers = []

        for ans in error_answers:
            file_name, line_offset, last_ans_position, content_inputs, prediction, target = ans

            # Check if the prediction is in the correct list
            if prediction in self.correct_list:
                content_inputs_tokens = content_inputs[last_ans_position:].split()
                target_tokens = target.split()

                if content_inputs_tokens and target_tokens:
                    corrected_words = []
                    idx = 0

                    # Find the best match for each target token
                    for target_token in target_tokens:
                        _, idx = self.find_best_match(content_inputs_tokens[idx:], [target_token])
                        best_match = content_inputs_tokens[idx]
                        corrected_words.append(best_match)

                    corrected_word = " ".join(corrected_words)
                    corrected_word = re.sub(r'[^A-Za-z0-9,\s.-]', '', corrected_word)

                    _, _, answer_format = self.calculate_answer_offset(file_name, line_offset, last_ans_position, content_inputs, prediction, corrected_word)

                    # Add corrected answer to the list
                    if answer_format is not None:
                        idx = list(map(int, answer_format.split('\t')[2:4]))
                        if idx not in idx_buffer:
                            corrected_answers.append(answer_format)
                            idx_buffer.append(idx)
                            print(f'{answer_format} (corrected)')
                        else:
                            print(f'{corrected_word} (correction error)')
            else:
                print(f'Not found: {content_inputs}')

        return corrected_answers

    def find_best_match(self, list1, list2):
        # Encode the lists of tokens into sentence embeddings
        embeddings1 = self.sentence_transformer.encode(list1, convert_to_tensor=True).cpu()
        embeddings2 = self.sentence_transformer.encode(list2, convert_to_tensor=True).cpu()

        # Create an index and add the embeddings of the first list to it
        index = faiss.IndexFlatL2(embeddings1.size(1))
        index.add(embeddings1.numpy())

        # Search for the nearest neighbors in the index for the embeddings of the second list
        D, I = index.search(embeddings2.numpy(), 1)

        # Return the distance and index of the best match
        return D[0][0], I[0][0]
    """

   
