import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from .maps import LABEL_MAP, OTHER_LABEL

MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512

class VulnerabilityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.code = df['code'].to_numpy()
        self.labels = df['label_id'].to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.code)

    def __getitem__(self, item):
        code_str = str(self.code[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            code_str,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_label_id(row):
    # Caso Safe
    if row['cwe_id'] == 'Safe':
        return 0
    # Si es vulnerable, buscamos el ID en tu mapa
    cwe = row['cwe_id']
    return LABEL_MAP.get(cwe, OTHER_LABEL)

def create_data_loaders(csv_path, batch_size=8, test_size=0.1):
    print(f"[INFO] Cargando dataset desde {csv_path}...")
    
    # Leemos solo columnas de codigo y el target
    df = pd.read_csv(csv_path, usecols=['code', 'cwe_id'])
    df = df.dropna(subset=['code'])
    
    print("[INFO] Generando etiquetas...")
    df['label_id'] = df.apply(get_label_id, axis=1)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    val_len = int(len(df) * test_size)
    train_df = df.iloc[:-val_len]
    val_df = df.iloc[-val_len:]
    
    print(f"[INFO] Train Size: {len(train_df)} | Val Size: {len(val_df)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = VulnerabilityDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = VulnerabilityDataset(val_df, tokenizer, MAX_LEN)
    
    # Num workers optimizado para Windows/Linux
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader