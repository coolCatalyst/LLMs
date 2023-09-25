import copy
import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_name = "microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 0   # unk. we want this to be different from the eos token pad_token = '!'
tokenizer.padding_side = "right"  # Allow batched inference

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True,
)
base_model.train()
model = copy.deepcopy(base_model)
# base_model.requires_grad_ = True
for name, param in model.named_parameters():
    param.requires_grad = True

from datasets import load_dataset
cutoff_len = 256
# train_on_inputs = True

def generate_prompt(data_point):
  return f"""
Human: {data_point["user"]}
AI: {data_point["AI"]}
  """.strip()

def generate_and_tokenize_prompt(data_point):
  full_prompt = generate_prompt(data_point)
  result = tokenizer(full_prompt, padding=True, truncation=True, max_length=cutoff_len) #, return_tensors=None)
  return result

dataset = load_dataset('json', data_files='qa_gpt4.json', split="train")
dataset = dataset.shuffle().map(generate_and_tokenize_prompt)

OUTPUT_DIR = "/root/hongyu/JupyterNotebooksFinetuning/models/phi1.5"
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    # warmup_ratio=0.5,
    # warmup_steps=5,
    auto_find_batch_size=True,
    num_train_epochs=5,
    learning_rate=2e-4,
    # fp16=True,
    # optim='adamw_torch',
    # bf16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir=OUTPUT_DIR,
    save_strategy='epoch',
    adam_epsilon=1e-6
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False) #, return_tensors='pt')  #, pad_to_multiple_of=8),
)


model.config.use_cache = False
trainer.train()

print()