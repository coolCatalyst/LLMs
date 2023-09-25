import re
import torch

prompt_template = """
{question}

Answer: 
"""

def phi_1_5_inference(model, tokenizer, question:str, max_length:int=512):
    """
    Generate the inference for a question using the Phi 1.5 model.

    Parameters:
        model (torch.nn.Module): The Phi 1.5 model for inference.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        question (str): The question to generate inference for.
        max_length (int, optional): The maximum length of the generated text. Defaults to 512.

    Returns:
        str: The inferred answer for the given question.
    """
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
