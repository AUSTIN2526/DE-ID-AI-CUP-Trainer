from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import numpy as np
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def same_seeds(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def use_accelerator(*args, **kwargs):
    # Fully Sharded Data Parallel
    # https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    
    return accelerator.prepare(args)[0] if len(args) == 1 else accelerator.prepare(args)


def merge_model(base_model_id, adapter_name):
    
    # 推理時自動合併模型權重，因使用QLoRA在前向傳播時會多計算一個旁路分支，導致推理速度變慢
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        device_map={"": "cpu"}, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(
        base_model, 
        f'adapter/{adapter_name}', 
        torch_dtype=torch.bfloat16, 
        device_map={"": "cpu"},
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )

    model = model.merge_and_unload()
    model.save_pretrained(f'model/{adapter_name}')
    tokenizer.save_pretrained(f'model/{adapter_name}')

