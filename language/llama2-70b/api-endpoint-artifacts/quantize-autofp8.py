from datasets import load_dataset
from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "Llama-2-70b-chat-hf"
quantized_model_dir = "Llama-2-70b-chat-hf-fp8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Load and tokenize all dataset samples for calibration of activation scales
ds = load_dataset("mgoin/ultrachat_2k", split="train_sft")
examples = [tokenizer.apply_chat_template(batch["messages"], tokenize=False) for batch in ds]
examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt", max_length=4096).to("cuda")
print(examples)

# Define quantization config with static activation scales
quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="static",
    ignore_patterns=["re:.*lm_head"]
)

# Load the model, quantize, and save checkpoint
model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
