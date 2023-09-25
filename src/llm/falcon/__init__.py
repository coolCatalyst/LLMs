import torch
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain import PromptTemplate, LLMChain
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model_id = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

default_template = """
Human: {input} 
AI:"""

def get_chain(model, template, verbose=False):
    """
    Generates a language model chain using a given model and template.
    
    Args:
        model (str): The name of the language model to use.
        template (str): The template to use for generating the language model chain.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    
    Returns:
        LLMChain: The generated language model chain.
    """
    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        stop_sequence="\nAI:",
        temperature=0.7,
        max_new_tokens=512,
        repetition_penalty=1.2,
    )
    return LLMChain(
        llm=HuggingFacePipeline(pipeline=pipeline),
        prompt=PromptTemplate.from_template(template),
        verbose=verbose,
    )

def falcon_inference(model_path:str, question:str):
    """
    Perform inference using the Falcon model.

    Args:
        model_path (str): The path to the directory containing the Falcon model.
        question (str): The input question for the inference.

    Returns:
        str: The result of the Falcon inference.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config
    )

    chain = get_chain(model, template=default_template, verbose=False)
    res = chain.run(question)
    return res