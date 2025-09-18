from utils.load_model_tokenizer import load_model_tokenizer
from utils.build_tool import same_seeds, merge_model
from utils.preprocess_utils import sft_format
from utils.dataset import DatasetLoader, open_file
from utils.inference import TextInference

import os 
from config import config



    
args = config()
same_seeds(args.seed)
if args.adapter_name not in os.listdir('model'):
    merge_model(
        'Qwen/Qwen-14B' if args.data_type !='Time' else 'Qwen/Qwen-7B', 
        args.adapter_name
    )
    
model, tokenizer = load_model_tokenizer(
    f'model/{args.adapter_name}',
    use_QLoRA=False
)
model.eval()
model.config.use_flash_attn = True
Loader = DatasetLoader(
    data_dir=args.valid_file_dir,
)


test_file = Loader(
    data_type=args.data_type,
    mode='Valid',
)

inference = TextInference(
    model = model, 
    tokenizer = tokenizer, 
    output_directory = args.out_dir, 
    window_limit = args.windows_size,
    prompt = open_file(args.prompt_path), 
    sft_formatter = sft_format,
    data_type = args.data_type
)

inference(test_file, sliding = True if args.data_type != 'Time' else False)
