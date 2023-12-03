from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_model_tokenizer(base_model_id, use_QLoRA = True):
    
    # 不支援全量與多卡微調，use_QLoRA參數是為了測試經度損失狀態下的結果，與推理之用
    

    # 讀取Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        add_special_tokens=False
    )

    # PAD token可以隨機使用
    if not tokenizer.eos_token:
        tokenizer.eos_token = list(tokenizer.special_tokens.keys())[0]
        print(f'自動設定 eos token {tokenizer.eos_token}')

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'自動設定 pad token {tokenizer.eos_token}')

    # 模型設定
    device_map = {"": Accelerator().local_process_index}
    torch_dtype = torch.bfloat16 # 部分電腦不支援

    # 量化設定
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype
    )
   # 讀取模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,     
        device_map=device_map, 
        use_cache=False,
        trust_remote_code=True,
        use_flash_attn = False,
        bf16 = True
    )
    
    # 論文網址:https://arxiv.org/abs/2305.14314
    if use_QLoRA:
        peft_config = LoraConfig(
            r=32,        
            target_modules=["c_attn"],
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )

        # 使用prepare_model_for_kbit_training後不需進行自行適配float32與開啟embedding層操作
        # https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py
        # 此區段預設本來就為True但有可能會有版本問題所以自定義參數防呆
        gradient_checkpointing_kwargs={'use_gradient_checkpointing':True}
        model = prepare_model_for_kbit_training(model, **gradient_checkpointing_kwargs)
        model = get_peft_model(model, peft_config) 


    return model, tokenizer
