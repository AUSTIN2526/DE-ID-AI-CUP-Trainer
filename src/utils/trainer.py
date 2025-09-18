import torch
import torch.optim as optim
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.modeling_utils import unwrap_model
from tqdm import tqdm
import matplotlib.pyplot as plt 


def get_optimizer_scheduler(model, epochs, data_len, lr=5e-5, warmup_ratio=0.2):
    total_steps = data_len * epochs                  
    warmup_steps = int(data_len * warmup_ratio)    

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps, 
        num_cycles=1, 
    )

    return optimizer, scheduler

class Trainer:
    def __init__(
        self, model, epochs, train_loader, valid_loader,
        optimizer, scheduler, early_stopping=2,
        neftune_noise_alpha=5,
        file_name='adapter', neftune=True
    ):

        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.neftune_noise_alpha = neftune_noise_alpha
        self.file_name = file_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 是否啟用NEFtune
        if neftune:
            self.model = self.activate_neftune(model)
        else:
            self.model = model
        
    def train_epoch(self, epoch):
        train_loss = 0
        train_pbar = tqdm(self.train_loader, position=0, leave=True)
        
        self.model.train() 
        for batch_cnt, input_datas in enumerate(train_pbar): 
            step = epoch * len(self.train_loader) + batch_cnt
    
            input_datas = {k: v.to(self.device) for k, v in input_datas.items()}
                
            self.optimizer.zero_grad() 
            outputs = self.model(**input_datas)
            
            loss = outputs[0]
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            
            loss.backward()
            self.optimizer.step() 
            self.scheduler.step(step)
            
            train_pbar.set_description(f'Train Epoch {epoch}') 
            train_pbar.set_postfix({'loss':f'{loss.item():.3f}', 'lr' :f'{format(lr, ".3e")}'})
    
            train_loss += loss.item()  
    
        return train_loss / len(self.train_loader) 

    def validate_epoch(self, epoch):
        valid_loss = 0
        valid_pbar = tqdm(self.valid_loader, position=0, leave=True)
        
        self.model.eval()
        with torch.no_grad():
            for input_datas in valid_pbar:
                input_datas = {k: v.to(self.device) for k, v in input_datas.items()}
            
                outputs = self.model(**input_datas) 
                loss = outputs[0]
                
                valid_pbar.set_description(f'Valid Epoch {epoch}')
                valid_pbar.set_postfix({'loss':f'{loss.item():.3f}'})
    
                valid_loss += loss.item()
    
        return valid_loss / len(self.valid_loader)

    def train(self, show_loss=True):
        best_loss = float('inf')
        loss_record = {'train': [], 'valid': []}
        stop_cnt = 0
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss = self.validate_epoch(epoch)
    
            loss_record['train'].append(train_loss)
            loss_record['valid'].append(valid_loss)

            # Save the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                save_path = f'adapter/{self.file_name}_{epoch}'
                self.model.save_pretrained(save_path)
                print(f'Saving Model With Loss {best_loss:.5f}')
                stop_cnt = 0
            else:
                stop_cnt += 1

            # Early stopping
            if stop_cnt == self.early_stopping:
                output = "Model can't improve, stop training"
                print('-' * (len(output) + 2))
                print(f'|{output}|')
                print('-' * (len(output) + 2))
                break

            print(f'Train Loss: {train_loss:.5f}', end='| ')
            print(f'Valid Loss: {valid_loss:.5f}', end='| ')
            print(f'Best Loss: {best_loss:.5f}', end='\n\n')

        if show_loss:
            self.show_training_loss(loss_record)
        self.de_neftune(self.model)
    
    def show_training_loss(self, loss_record):
        train_loss, valid_loss = [i for i in loss_record.values()]
        
        plt.plot(train_loss)
        plt.plot(valid_loss)
        # Title
        plt.title('Result')
        # Y-axis label
        plt.ylabel('Loss')
        # X-axis label
        plt.xlabel('Epoch')
        # Display line names
        plt.legend(['train', 'valid'], loc='upper left')
        # Display the line chart
        plt.show()
        
    def de_neftune(self, model):
        # 移除hook (若程式有需要繼續進行運算，但在該程式中並沒有因此可有可無)
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.base_model.model.get_input_embeddings() # 這是LoRA的Embedding層
        #embeddings = unwrapped_model.get_input_embeddings() # 沒有LoRA的
        self.neftune_hook_handle.remove()

        del embeddings.neftune_noise_alpha

            
    def activate_neftune(self, model):
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.base_model.model.get_input_embeddings() # 這是LoRA的Embedding層
        #embeddings = unwrapped_model.get_input_embeddings() # 沒有LoRA的
        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        # hook embedding layer
        hook_handle = embeddings.register_forward_hook(self.neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        
        return model
        
    def neftune_post_forward_hook(self, module, input, output):
        # 公式來源:https://github.com/neelsjain/NEFTune
        # 論文網址:https://arxiv.org/abs/2310.05914
        if module.training:
            dims = torch.tensor(output.size(1) * output.size(2))
            mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
            output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
                
        return output
    


