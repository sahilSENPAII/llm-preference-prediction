from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-large"
    num_labels: int = 3

    max_length: int = 1024

    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True

    train_path: str = "/kaggle/input/lmsys-chatbot-arena/train.csv"
    test_path: str = "/kaggle/input/lmsys-chatbot-arena/test.csv"
    output_dir: str = "/kaggle/working/output"
    val_split: float = 0.1

    seed: int = 42
