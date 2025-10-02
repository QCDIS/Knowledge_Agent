"""
https://cdi.seadatanet.org/report/5/json
"""



from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer  # Hugging Face's SFT Trainer for fine-tuning
from peft import LoraConfig  # LoRA for efficient fine-tuning
from transformers import AutoModelForCausalLM

# Load a tokenizer
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
base_model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding

# Raw text data
raw_text = """
Data set name: Wind and wave data from North Sea Platforms (1974-1987)
Data holding centre: United Kingdom Offshore Operators Association
Country: United Kingdom
Time period: Various periods between 1974 and 1987
Ongoing: No
Geographical area: North Sea
Observations:
    Parameters: Wind strength and direction; Wave direction; Spectral wave data parameters; Wave height and period statistics
    Instruments: Anemometers; wave recorders
Description:
    Summary: The data set comprises various measurements of winds and waves, mostly collected by Marex (now Paras), on behalf of UKOOA. Wind data from Brent Platform and wind and wave data from North Cormorant were gathered by Shell. 
Availability:
    Organisation: United Kingdom Offshore Operators Association
    Availability: By negotiation
    Contact: The Director
    Address: United Kingdom Offshore Operators Association 3 Hans Crescent London SW1X 0LN United Kingdom
    Telephone: +44 171 589 5255
Administration:
    Collating centre: British Oceanographic Data Centre
    Local identifier: 1089002
    Global identifier: 854
    Last revised: 2009-10-15
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


# Convert into a Hugging Face dataset
dataset = Dataset.from_dict({"text": [raw_text]})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# LoRA Configuration for parameter-efficient fine-tuning
peft_config = LoraConfig(
    r=8,  # Rank of LoRA matrix (lower = less memory usage)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout for regularization
    task_type="SEQ_2_SEQ_LM",  # Task type: Causal Language Modeling
)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,  # Load model in 4-bit precision for lower memory usage
    device_map="auto"  # Automatically assign layers to GPUs
)

training_args = TrainingArguments(
    output_dir="./output_model",
    per_device_train_batch_size=2,  # Adjust batch size based on memory
    gradient_accumulation_steps=8,  # Helps with large models
    num_train_epochs=100,  # Number of epochs
    logging_steps=10,  # Log every 10 steps
    save_strategy="epoch",  # Save model at the end of every epoch
    learning_rate=2e-4,  # Fine-tuning learning rate
    evaluation_strategy="no",  # No evaluation for this example
    save_total_limit=2,  # Keep only last 2 checkpoints
    fp16=True,  # Enable mixed-precision training
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    peft_config=peft_config,  # Use LoRA for efficient fine-tuning
)

# Start Training
trainer.train()

trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")




