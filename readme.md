# Milind's LLM Models Demo

This Jupyter notebook provides a demo for loading and using large language models (LLMs) like GPT-3 for text generation.

## Requirements

The notebook requires the following packages:

- transformers
- huggingface_hub
- peft
- torch
- einops
- auto_gptq


## Contents

The notebook does the following:

1. Installs required packages
2. Loads a pretrained GPT-3 model from HuggingFace Hub  
3. Loads a PEFT model to add extra embeddings
4. Defines a text generation function 
5. Generates text for some sample prompts

The model used is a 40B parameter GPT-3 model fine-tuned with reinforcement learning.



# Script to scrape data
To scrap the data from discord use the script provided in this github
Make sure you use your autorization discord id which can be taken by going to developer tools in chrome after opening going to chrome 
And then enable developer tools in discord and copy channel id 
add these to script and run 
at end you will get a csv file

