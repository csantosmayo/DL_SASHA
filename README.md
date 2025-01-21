# SASHA
DL SASHA Project for Steerable LLMs

## Files
In this repo you will find the following files:
* main_unsloth.ipynb: notebook to begin/resume LLM training and save model checkpoints
* test_losses_unsloth.ipynb: notebook to compute the test losses for a trained LLM
* main_inference.ipynb: notebook for LLM text generation on a subset of the test set
* main_evaluation.ipynb: notebook that uses GPT-4o (OpenAI API calls) to compute the win rate between two provided models (baseline vs SASHA).
* utils.py: python file for general data loading, handling and model saving and loading

## Reproducibility
To reproduce the results use the files above as it is described next:
* **To train a model**: choose the model name, saving path and resume option to initialize training. Further, define details like LoRA configuration, chosen attributes, number of attributes per batch and initial probabilities for your model.
* **To compute the test losses**: select the pretrained model (set the loading path and resume options accordingly), load the test dataset, choose the testing configuration (attributes, probabilties and attributes per batch).
* **To run text generation**: select the pretrained model you want for text generation (set the loading path and resume options accordingly), load the test dataset, choose the number of attributes to include in the prompt and  
* **To run model evaluation with GPT-4o**: load baseline responses and SASHA responses (run the loading once for each model from their loading directories). Use the OpenAI API to compute win rates and ties of the responses.
