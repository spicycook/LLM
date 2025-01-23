# LLM
I put some LLM scripts that I would love to share publicly


In Step 1: I finetuned a Llama3 8B pretrained model (note that this is NOT instruct model), using an Nvidia A100 80GB GPU. The model is 8bit quantized. I used AUC as my performsnce metrics since my labels are 1 and 0. After finetuning, I saved adapters rather than the whole model.

In Step 2: I demonstrated how to load the finetuned adpaters and add them into a base Llama3 8B pretrained model. 

