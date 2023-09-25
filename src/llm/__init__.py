from IPython.display import display, Markdown
from langchain import HuggingFacePipeline, LLMChain, PromptTemplate
import transformers

default_template = """
Human: {input} 
AI:"""


def print_num_params(model):
    """
    Print the number of parameters in the model.

    Parameters:
        model (nn.Module): The model whose parameters will be counted.

    Returns:
        None
    """
    params = [
        (param.numel(), param.numel() if param.requires_grad else 0)
        for _, param in model.named_parameters()
    ]
    all, train = map(sum, zip(*params))
    print(f"{train=} / {all=} {train/all:f}")


def get_chain(model, tokenizer, template=default_template, verbose=False):
    """
    Generate the LLMChain object to generate text using the specified model and tokenizer.
    
    Args:
        model (str): The name of the pre-trained model to use.
        tokenizer (str): The name of the tokenizer to use.
        template (str, optional): The template to use for generating text. Defaults to the default template.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    
    Returns:
        LLMChain: The generated LLMChain object.
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

