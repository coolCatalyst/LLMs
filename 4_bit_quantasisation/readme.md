# 4 Bit Quantisation
The script given in this folder is used to 4 bit quantize the weight and parametrs using GPTQ algotithm using hugging face
After converting the weights to 4bit we need to train the model to finetune the new weights as written in the paper we can do this by using some default ans standard datasets like c4 wiki4pedia (recommended) but we can do this with our own dataset which we have implemented in the given script(but we have to make sure the data is given as list of strings)

This helps us to lower the ram requirement (64 bit floating to 16 bit floating numbers) 
This is a much needed step for training the model on consumer GPU

