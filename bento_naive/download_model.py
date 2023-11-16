import transformers
from transformers import AutoTokenizer
import bentoml
import torch

model = "beomi/llama-2-ko-7b"
tokenizer = AutoTokenizer.from_pretrained(model)
task = "text-generation"

bentoml.transformers.save_model(
    task,
    transformers.pipeline(
    task,
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    ),
    metadata=dict(model_name=model),
)

