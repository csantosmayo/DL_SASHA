import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
from transformers import LlamaForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset, Dataset, load_from_disk
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from unsloth import FastLanguageModel

def sample_selector(dataset, max_length):
    """ 
    Selects samples from the dataset based on the length of the prompt and response
    """
  
    indices = []
    for i in range(len(dataset)):
        if dataset[i]["length"] <= max_length:
            indices.append(i)

    return dataset.select(indices)


def tokenization(tokenizer, batch, features):
    """ 
    Tokenizes the batch of samples based on the features provided
    Args:
      tokenizer: The tokenizer to be used
      batch: The batch of samples
      features: The features to be tokenized

    Returns:
      tokenized: The tokenized dictionary
    """

    # Returns tokenized dictionary
    tokenized = {}

    for feature in features: # prompt/response/helpfulness/correctness/coherence...
        if feature == 'prompt':
            texts = batch[feature]
            tokens = tokenizer(texts, padding = False, add_special_tokens=False)
            tokenized[f"{feature}_labels"] = [[-100] * len(ids) for ids in tokens["input_ids"]] # Labels

        elif feature == 'response':
            texts = batch[feature]
            tokens = tokenizer(texts, padding = False, add_special_tokens=False)
            tokenized[f"{feature}_labels"] = tokens["input_ids"] # Labels

        else:
            texts = [f"[{feature}:{x}] " for x in batch[feature]]
            tokens = tokenizer(texts, padding = False, add_special_tokens=False)
            tokenized[f"{feature}_labels"] = [[-100] * len(ids) for ids in tokens["input_ids"]] # Labels

        for key, value in tokens.items():
            tokenized[f"{feature}_{key}"] = value # Ids, attention_mask

    return tokenized


class dataset_concatenator:
  """
  Class to concatenate the prompt and response samples
  """

  def __init__(self, bos_token_id, eos_token_id, dict_instruct, attributes_to_use):
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id
    self.dict_instruct = dict_instruct
    self.attributes_to_use = attributes_to_use

  # mapping function
  def map(self, data):

    # Get user and response instruction dictionary lengths
    instruct_length = self.dict_instruct['user_labels'].shape[0] + self.dict_instruct['response_labels'].shape[0]
    data['prompt_answer_ids'] = torch.cat([self.dict_instruct['user_input_ids'],
                                            data['prompt_input_ids'],
                                            torch.tensor(self.eos_token_id).unsqueeze(dim = 0),
                                            self.dict_instruct['response_input_ids'],
                                            data['response_input_ids'],
                                            torch.tensor(self.eos_token_id).unsqueeze(dim = 0)],
                                           dim = 0)
    data['prompt_answer_att_mask'] = torch.cat([self.dict_instruct['user_attention_mask'],
                                                data['prompt_attention_mask'],
                                                torch.tensor(1).unsqueeze(dim = 0),
                                                self.dict_instruct['response_attention_mask'],
                                                data['response_attention_mask'],
                                                torch.tensor(1).unsqueeze(dim = 0)],
                                               dim = 0)
    data['prompt_answer_labels'] = torch.cat([self.dict_instruct['user_labels'],
                                              data['prompt_labels'],
                                              torch.tensor(-100).unsqueeze(dim = 0),
                                              self.dict_instruct['response_labels'],
                                              data['response_labels'],
                                              torch.tensor(self.eos_token_id).unsqueeze(dim = 0)],
                                             dim = 0)
    data['length'] = instruct_length + data['prompt_labels'].shape[0] + data['response_labels'].shape[0] + 2

    return data


  def __call__(self, dataset):
    return dataset.map(self.map, batched=False, remove_columns=['prompt_input_ids', 'prompt_attention_mask', 'prompt_labels',
                                                                'response_input_ids', 'response_attention_mask', 'response_labels'])
  

class AttributeCollate:
    """
    Class to collate the samples in the batch
    """

    def __init__(self, attributes, attribute_probs, num_attributes_per_batch, dict_instruct, bos_id, eos_id):
        self.attributes = attributes
        self.attribute_probs = attribute_probs
        self.num_attributes_per_batch = num_attributes_per_batch
        self.dict_instruct = dict_instruct
        self.bos_id = bos_id
        self.eos_id = eos_id

    def set_attributes(self, new_attributes):
        self.attributes = new_attributes

    def set_num_attributes_per_batch(self, new_num):
        self.num_attributes_per_batch = new_num

    def set_attribute_probs(self, new_probs):
        self.attribute_probs = new_probs


    def __call__(self, batch):

        # batch is a list of samples, each a dict of tensors
        # Ensure attributes are selected without repetition
        if self.num_attributes_per_batch > len(self.attributes):
            raise ValueError("num_attributes_per_batch cannot exceed the number of unique attributes available.")
        
        # Normalize attribute probs
        self.attribute_probs = self.attribute_probs / np.sum(self.attribute_probs)

        # Numpy sampling
        if self.num_attributes_per_batch == len(self.attributes):
            visible_attributes = self.attributes
        else:
            visible_attributes = np.random.choice(np.arange(0, len(self.attributes)), size = self.num_attributes_per_batch, replace = False, p=self.attribute_probs)
            visible_attributes = [self.attributes[i] for i in visible_attributes]


        # bos + 'System Prompt: Answer the following user isntruction based on the provided alignment attributes. '
        system_instruct_ids = torch.stack([self.dict_instruct['system_input_ids']] * len(batch), dim=0)
        system_instruct_attmask = torch.full(system_instruct_ids.shape, 1)
        system_instruct_labels = torch.full(system_instruct_ids.shape, -100)

        # Actual attributes - 1 stack per attribute
        attr_id_list = []
        attr_attmask_list = []
        for attr in visible_attributes:

            single_attr_ids = [sample[f"{attr}_input_ids"] for sample in batch]
            single_attr_attmask = [sample[f"{attr}_attention_mask"] for sample in batch]

            single_attr_ids = torch.stack(single_attr_ids, dim=0)             # [batch_size, attr_seq_len]
            single_attr_attmask = torch.stack(single_attr_attmask, dim=0)     # [batch_size, attr_seq_len]

            attr_id_list.append(single_attr_ids)
            attr_attmask_list.append(single_attr_attmask)

        # Multiple attributes are concatenated along the sequence dimension (dim=1)
        if len(attr_id_list) > 0:
            attribute_ids = torch.cat(attr_id_list, dim=1)              # [batch_size, sum_of_all_attr_seq_len]
            attribute_attmask = torch.cat(attr_attmask_list, dim=1)     # [batch_size, sum_of_all_attr_seq_len]
            attribute_labels = torch.full(attribute_ids.shape, -100)
        else:
            # If no attributes selected, just empty tensors
            attribute_ids = torch.empty(0, dtype=torch.long)
            attribute_attmask = torch.empty(0, dtype=torch.long)
            attribute_labels = torch.empty(0, dtype=torch.long)

        # 'User Instruction: ' + prompt + (eos + 'Response: ') + answer + eos
        # Get the maximum
        max_sequence_length = 0
        for sample in batch:
          if sample['length'] > max_sequence_length:
            max_sequence_length = sample['length']

        # Init
        pad_prompt_answer_ids = torch.full((len(batch), max_sequence_length), self.eos_id)
        pad_prompt_answer_att_mask = torch.zeros((len(batch), max_sequence_length), dtype=torch.long)
        pad_prompt_answer_labels = torch.full((len(batch), max_sequence_length), -100)


        for i in range(len(batch)):
          pad_prompt_answer_ids[i, :len(batch[i]['prompt_answer_ids'])] = batch[i]['prompt_answer_ids']
          pad_prompt_answer_att_mask[i, :len(batch[i]['prompt_answer_att_mask'])] = batch[i]['prompt_answer_att_mask']
          pad_prompt_answer_labels[i, :len(batch[i]['prompt_answer_labels'])] = batch[i]['prompt_answer_labels']

        # Horizontal concatenation: # bos + 'System Prompt' + 'Alignment attributes: ' + actual attributes + 'User Instruction: ' + prompt + (eos + 'Response: ') + answer + eos
        input_ids = torch.cat([system_instruct_ids, attribute_ids, pad_prompt_answer_ids], dim=1)
        attention_mask = torch.cat([system_instruct_attmask, attribute_attmask, pad_prompt_answer_att_mask], dim=1)
        labels = torch.cat([system_instruct_labels, attribute_labels, pad_prompt_answer_labels], dim=1)

        # Inputs for model
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'attributes': visible_attributes
        }

        return inputs
    

class AlignmentTrainer:
  """
  Class to train the alignment model
  """

  def __init__(self, alpha, beta, T, attributes, attribute_probs, num_attributes_per_batch):
    self.alpha = alpha  # parameter for combined loss computation
    self.beta = beta    # parameter for momentum
    self.T = T          # temperature parameter for softmax
    self.attributes = attributes
    self.attribute_probs = np.array(attribute_probs, dtype=float)
    self.num_attributes_per_batch = num_attributes_per_batch

  def compute_momentum(self, new_probs):
    new_probs = self.beta * self.attribute_probs + (1 - self.beta) * new_probs
    new_probs = new_probs / new_probs.sum()
    self.attribute_probs = new_probs
    return new_probs

  def compute_new_probs(self, train_losses, val_losses):
    # convert train losses and val losses from dict to numpy array
    train_losses = np.array(list(train_losses.values()))
    val_losses = np.array(list(val_losses.values()))

    # compute the combined losses (numpy array)
    combined_losses = self.alpha * train_losses + (1 - self.alpha) * val_losses

    # compute the softmax of the combined losses
    softmax_losses = np.exp(combined_losses / self.T) / np.sum(np.exp(combined_losses / self.T))

    # compute the new probabilities using momentum
    new_probs = self.compute_momentum(softmax_losses)

    return new_probs.tolist()
    

def save_checkpoint(save_dir, epoch, model, training_dict, optimizers, train_loss, val_loss):
    """
    Save the LoRA weights, optimizer state, and scheduler state from a single folder for the given epoch.

    Args:
        save_dir (str): Path to the environment folder.
        model (torch.nn.Module): The base model of which to save LoRA weights.
        training_dict (dict): Dictionary containing epoch, attributes, probabilities, and num_attributes_per_batch.
        learning_rates (dict): Dictionary containing attributes keys, int values for each attribute.
        optimizers (dict): Dictionary containing attribute keys, torch.optim.Optimizer values for each attribute.
        schedulers (dict): Dictionary containing attribute keys, torch.optim.lr_scheduler values for each attribute.
        train_loss (dict):
        val_loss (dict):


    Returns:
        None
    """

    # Create a folder for the current epoch
    epoch_folder = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)

    # Save the LoRA weights in the epoch folder
    model.save_pretrained(epoch_folder + "/lora_model")
    # model.save_pretrained(epoch_folder, safe_serialization = False, save_peft_format = False)

    # Save the optimizer and scheduler states in the same epoch folder
    training_dict['epoch'] = epoch

    checkpoint = {
        "training_dictionary": training_dict,
        "optimizer_state_dict": {key: optimizer.state_dict() for key, optimizer in optimizers.items()},
        # "scheduler_state_dict": {key: scheduler.state_dict() for key, scheduler in schedulers.items()},
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    optimizer_checkpoint_path = os.path.join(epoch_folder, "optimizer_scheduler.pt")
    torch.save(checkpoint, optimizer_checkpoint_path)

    print(f"All checkpoint data for epoch {training_dict['epoch']} saved in {epoch_folder}.")


def load_checkpoint(save_dir, epoch, model):
    """
    Load the LoRA weights, optimizer state, and scheduler state from a single folder for the given epoch.

    Args:
        save_dir (str): Path to the environment folder.
        epoch (int): Epoch save of which to load the weights.
        model (torch.nn.Module): The model with already loaded LoRA weights.

    Returns:
        training_dict (dict): Dictionary containing epoch, attributes, probabilities, and num_attributes_per_batch.
        optimizers (dict): Dictionary containing attribute keys, torch.optim.Optimizer values for each attribute.
        lr_schedulers (dict): Dictionary containing attribute keys, torch.optim.lr_scheduler values for each attribute.
    """

    # Fetch the epoch folder
    epoch_folder = os.path.join(save_dir, f"epoch_{epoch}")

    # Load checkpoint
    states_path = os.path.join(epoch_folder, "optimizer_scheduler.pt")
    checkpoint = torch.load(states_path)

    # Load training dictionary
    training_dict = checkpoint["training_dictionary"]

    optimizer_keys = checkpoint["optimizer_state_dict"].keys()

    # Instantiate optimizer and scheduler
    optimizers = {key: AdamW(model.parameters(), lr=1e-6, weight_decay=1e-2) for key in optimizer_keys}
    # lr_schedulers = {key: CyclicLR(optimizers[key], base_lr=1e-6, max_lr=5e-6, step_size_up=2000) for key in optimizers}

    # lr_schedulers = {key: get_scheduler(
    #   name=optimizer_name,
    #   optimizer=optimizers[key],
    #   num_warmup_steps=50,
    #   scheduler_specific_kwargs = {"last_epoch": epoch},
    # ) for key in optimizers}

    # Load optimizer and scheduler state
    for key in optimizers:
      optimizers[key].load_state_dict(checkpoint["optimizer_state_dict"][key])
      # lr_schedulers[key].load_state_dict(checkpoint["scheduler_state_dict"][key])

    return training_dict, optimizers


def custom_split(dataset, val_size):

    """
    Split dataset per pairs up to validation_size.

    Args:
        dataset (Dataset): The dataset to be split.
        val_size (int): The size of the validation dataset.

    Returns:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The validation dataset.

    Note:
        because in the SteerLM there are coupled instances of prompts (prompt 0 and 1 go together but have different alignment attributes and ground truth responses),
        we wanted to prevent data leakage into the testing set, which is why in this train-test partition, we keep the paired-prompts together (either both in train set or both in test set)
    """

    # Shuffle dataset indexes
    train_indexes = np.arange(len(dataset))
    np.random.shuffle(train_indexes)

    # Set of used indexes
    test_indexes = set()

    i = 0
    while len(test_indexes) < val_size:
         
        # sample new point and remove from possible list
        random_index = train_indexes[i]

        if random_index % 2 == 0: # If even, pair with next
            pair = random_index + 1
        else: # If odd, pair with previous
            pair = random_index - 1

        i += 1

        
        if random_index in test_indexes or pair in test_indexes:
            continue

        else:
            # Add to test_indexes
            test_indexes.update([random_index, pair])

    # Convert train and test_indexes to list
    train_indexes = list(set(train_indexes) - test_indexes)
    test_indexes = list(test_indexes)

    # Check
    shared_elements = list(set(train_indexes) & set(test_indexes))  # Convert back to a list
    print("Shared elements:", shared_elements)

    # Get the train and test datasets
    train_dataset = dataset.select(train_indexes)
    test_dataset = dataset.select(test_indexes)

    print('Check')
    print(train_indexes)
    print(test_indexes)

    return train_dataset, test_dataset


def set_datasets(dir, tokenizer, instruct_dictionary_tokenized, max_length = 1500, attributes_to_use = ['helpfulness', 'coherence', 'verbosity', 'correctness', 'complexity']):


    if not os.path.exists(dir): # If the directory does not exist, create the dataset and save

        # Save 
        # # Set tokenizer values
        # pad_token = tokenizer.eos_token
        # bos_token = tokenizer.bos_token
        # eos_token = tokenizer.eos_token

    
        # HelpSteer2 dataset
        dataset = load_dataset("nvidia/HelpSteer2")

        train_dataset = dataset['train']
        val_dataset = dataset['validation']

        train_dataset, test_dataset = custom_split(train_dataset, len(val_dataset))

        print('Check 2')
        print(len(train_dataset))
        print(len(val_dataset))
        print(len(test_dataset))

        # Tokenize dataset
        features = [feature for feature in train_dataset.features]

        # Tokenizaion
        train_dataset = train_dataset.map(lambda batch: tokenization(tokenizer, batch, features), batched=True, batch_size=32)
        val_dataset = val_dataset.map(lambda batch: tokenization(tokenizer, batch, features), batched=True, batch_size=32)
        test_dataset = test_dataset.map(lambda batch: tokenization(tokenizer, batch, features), batched=True, batch_size=32)

        # Converting to torch
        train_dataset.set_format(type="torch", columns=[f"{feature}_input_ids" for feature in features] +
                                                [f"{feature}_attention_mask" for feature in features] +
                                                [f"{feature}_labels" for feature in features])
        val_dataset.set_format(type="torch", columns=[f"{feature}_input_ids" for feature in features] +
                                                [f"{feature}_attention_mask" for feature in features] +
                                                [f"{feature}_labels" for feature in features])
        test_dataset.set_format(type="torch", columns=[f"{feature}_input_ids" for feature in features] +
                                                [f"{feature}_attention_mask" for feature in features] +
                                                [f"{feature}_labels" for feature in features])
        
        # Concatenating elements: user input, prompt, EOS, response, answer, EOS
        # Mapping the dataset for preprocessing and lenght filtering
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        # print lengths
        print(f"Initial training size: {len(train_dataset)}")
        print(f"Initial validation size: {len(val_dataset)}")

        concatenator = dataset_concatenator(bos_token_id, eos_token_id, instruct_dictionary_tokenized, attributes_to_use)

        train_dataset = concatenator(train_dataset)
        val_dataset = concatenator(val_dataset)
        test_dataset = concatenator(test_dataset)

        # Sample selection
        max_length = max_length
        train_dataset = sample_selector(train_dataset, max_length)
        val_dataset = sample_selector(val_dataset, max_length)
        test_dataset = sample_selector(test_dataset, max_length)

        # # Train, valid, test split
        # test_size = len(val_dataset)
        # split_dataset = train_dataset.train_test_split(test_size=test_size, shuffle=True, seed = 42)
        # train_dataset = split_dataset['train']
        # test_dataset = split_dataset['test']

        # Save dictionary
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        train_dataset.save_to_disk(os.path.join(dir, "train_dataset"))
        val_dataset.save_to_disk(os.path.join(dir,"val_dataset"))
        test_dataset.save_to_disk(os.path.join(dir,"test_dataset"))

    else: # If it does exist, load
        
        # Load the dataset
        train_dataset = load_from_disk(os.path.join(dir, 'train_dataset'))
        val_dataset = load_from_disk(os.path.join(dir,'val_dataset'))
        test_dataset = load_from_disk(os.path.join(dir,'test_dataset'))


    return train_dataset, val_dataset, test_dataset
        









    
