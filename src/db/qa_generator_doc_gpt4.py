import os
import random
import re
import string
import time
from typing import List
import openai

from tqdm import tqdm
from dotenv import load_dotenv
from src.prompt.prompt_doc import prompt_template
from src.utils import find_word_in_string, openai_chat_completion
from src.utils.json_io import jdump


def post_process_gpt4_response(message):
    """
    A function to post-process the response from GPT4.
    
    Args:
        message (dict): The response message from GPT4.
        
    Returns:
        list: A list of question-answer pairs extracted from the response.
    """
    if message is None:
        return []
    raw_messages = re.split("###", message["content"])
    qas = []
    for raw_qa in raw_messages:
        # filter out too short or too long instructions
        if len(raw_qa.split()) <= 3 or len(raw_qa.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, raw_qa) for word in blacklist):
            continue
        matches = re.findall(r":\s(.*?)\n", raw_qa)
        if len(matches) == 2:
            qas.append({"Question": matches[0], "Answer": matches[1]})
    return qas


def generate_qa_doc(
    documents: List[str] = [],
    output_dir: str = "./",
    model_name: str = "gpt-4",
    # request_batch_size=3,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    """
    Generates a QA dataset based on documents using the GPT-4 model.

    Args:
        documents (List[str], optional): A list of documents to generate QA pairs from. Defaults to [].
        output_dir (str, optional): The output directory to save the generated QA pairs. Defaults to "./".
        model_name (str, optional): The name of the GPT-4 model to use. Defaults to "gpt-4".
        temperature (float, optional): The temperature parameter for the GPT-4 model. Defaults to 1.0.
        top_p (float, optional): The top-p parameter for the GPT-4 model. Defaults to 1.0.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    machine_qas = []
    decoding_args = {
        "temperature": temperature,
        "n": 1,
        "max_tokens": 3072,  # hard-code to maximize the length. the requests will be automatically adjusted
        "top_p": top_p,
        "stop": ["\n20", "20.", "20."],
    }

    for document in tqdm(documents):
        messages = [
            {"role": "user", "content": prompt_template.replace("{context}", document)}
        ]
        res_message = openai_chat_completion(
            messages=messages,
            model_name=model_name,
            # batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={
                "50256": -100
            },  # prevent the <|endoftext|> token from being generated
        )
        # results.append(choices)

        qas = post_process_gpt4_response(res_message)
        machine_qas.extend(qas)
    jdump(machine_qas, os.path.join(output_dir, "qa_gpt4.json"))
