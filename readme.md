# LLM Models Demo

## STEP1->CREATION OF DATASET
file->(Discord_Messge_extractor_Milind.py)
This script help us to make a csv file of chats from a any discord 
With the help of this we have make dataset to train our LLM 
This uses librabries like request json and csv

With the help of this we have made two datasets StartUP and Tech(2) which have been created by scraping chats from various tech and AI channels on discord

## STEP2->4BIT QUANTIZATION 
file->(4_bit_quantasisation/4_bit_usingGPQ_by_rohit.ipynb)
This script is used to quant the parameters to 4bit using GPTQ algorithm which reduces the RAM and VRAM requirements for training these models
After converting the parameters to smaller precision we have tuned these weights 
For tuning these weights we can use the datasets recommended in paper like c4 or wiki4pedia or we can tune even using custom datasets like we have did (but dataset should be pass as list of strings)
we have used transformer and GPTQ librabries from hugging face

In the file we have used very small model but this can work on any lamma or gpt model

## STEP->3 FINETUNING
file->(Untitled43.ipynb)
At last we have finetuned our model using PEFT(LoRA) for mail genertaion
This is a very RAM effective algorithm for finetuning 

After creation of proper datasets we can tune this for discord chat



THESE ALL UNITS WORK PROPERLY AS INDIVIDUAL UNITS BUT STILL WE HAVE TO WORK ON MAKING A PIPELINE SO THEY CAN WORK AS ONE UNIT 
AND ALSO HAVE TO FIGURE OUT HOW WE CAN DEPLOY THESE MODELS


## Requirements

The notebook requires the following packages:

- transformers
- huggingface_hub
- peft
- torch
- einops
- auto_gptq


The notebook does the following:

1. Installs required packages
2. Loads a pretrained GPT-3 model from HuggingFace Hub  
3. Loads a PEFT model to add extra embeddings
4. Defines a text generation function 
5. Generates text for some sample prompts

The model used is a 40B parameter GPT-3 model fine-tuned with reinforcement learning.


# 4 Bit Quantisation (Rohit)


The script given in this folder is used to 4 bit quantize the weight and parametrs using GPTQ algotithm using hugging face
After converting the weights to 4bit we need to train the model to finetune the new weights as written in the paper we can do this by using some default ans standard datasets like c4 wiki4pedia (recommended) but we can do this with our own dataset which we have implemented in the given script(but we have to make sure the data is given as list of strings)

This helps us to lower the ram requirement (64 bit floating to 16 bit floating numbers) 
This is a much needed step for training the model on consumer GPU




#

