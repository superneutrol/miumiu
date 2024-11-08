from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer 
import json 
import glob
import safetensors 
from safetensors import safe_open
from typing import Tuple 
import os 

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Final all the .safetensors files 
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # load them one by one in the tensor dictionary 
    tensors = {}
    for safetensor_file in safetensors_files: 
        with safe_open(safetensor_file, framework="pt", device="cpu") as f: 
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    
    # Load the model's config 
    with open(os.path.join(model_path, "config.josn", "r")) as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # create a model using the configuration
    model = PaliGemmaForConditionalGeneration(config=config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)