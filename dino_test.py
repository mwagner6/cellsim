from huggingface_hub import hf_hub_download
import json

# Test if you can access the files directly
config_path = hf_hub_download(
    repo_id="facebook/dinov3-vit7b16-pretrain-lvd1689m",
    filename="config.json"
)
print("Config downloaded to:", config_path)

with open(config_path) as f:
    config = json.load(f)
    print("Model type:", config.get("model_type"))
    print("Architectures:", config.get("architectures"))

# Also check preprocessor
preprocessor_path = hf_hub_download(
    repo_id="facebook/dinov3-vit7b16-pretrain-lvd1689m",
    filename="preprocessor_config.json"
)
print("\nPreprocessor downloaded to:", preprocessor_path)

with open(preprocessor_path) as f:
    preproc = json.load(f)
    print("Image processor type:", preproc.get("image_processor_type"))