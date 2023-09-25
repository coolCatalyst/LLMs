import os
import random
import re
import string
import time
import openai

from tqdm import tqdm
from dotenv import load_dotenv
from src import utils


def post_process_gpt4_response(message):
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
        if any(utils.find_word_in_string(word, raw_qa) for word in blacklist):
            continue
        # if raw_qa.startswith("Write a program"):
        #     continue
        # # filter those starting with punctuation
        # if raw_qa[0] in string.punctuation:
        #     continue
        # # filter those starting with non-english character
        # if not inst[0].isascii():
        #     continue
        matches = re.findall(r":\s(.*?)\n", raw_qa)
        if len(matches) == 2:
            qas.append({"user": matches[0], "AI": matches[1]})
    return qas


def generate_qa_doc(
    documents=[],
    output_dir="./",
    model_name="gpt-4",
    # request_batch_size=3,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    os.makedirs(output_dir, exist_ok=True)
    prompt_template = open("./prompt_doc.txt").read() + "\n"

    machine_qas = []
    decoding_args = {
        "temperature": temperature,
        "n": 1,
        "max_tokens": 3072,  # hard-code to maximize the length. the requests will be automatically adjusted
        "top_p": top_p,
        "stop": ["\n20", "20.", "20."],
    }

    request_start = time.time()
    for document in tqdm(documents):
        messages = [
            {"role": "user", "content": prompt_template.replace("{context}", document)}
        ]
        res_message = utils.openai_chat_completion(
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
    request_duration = time.time() - request_start
    utils.jdump(machine_qas, os.path.join(output_dir, "qa_gpt4.json"))


if __name__ == "__main__":
    with open("assets/financial_blog.txt", "r") as file:
        text = file.read()
    documents = utils.compute_documents(text, chunk_size=300, chunk_overlap=100)
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    generate_qa_doc(documents=documents)
