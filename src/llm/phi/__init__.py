import re
import torch

prompt_template = """
{question}

Answer: 
"""

def phi_1_5_inference(model, tokenizer, question:str, max_length:int=512):
    prompt = prompt_template.format(question=question).strip()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    inputs = inputs.to(device) 

    outputs = model.generate(**inputs, max_length=max_length)
    text = tokenizer.batch_decode(outputs)[0]

    matches = re.findall(r'\nAnswer: (.*?)\n\n', text)

    if matches:
        return matches[0]
    else:
        return "No answer found in the text string."
