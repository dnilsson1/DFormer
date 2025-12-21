from local_configs.JARVIS.DFormerv2_Base_custom import C as config

# Turn off pretrained model to skip loading missing pretrained weights just for evaluation in this container
config.pretrained_model = ""

# Optionally adjust eval sources if needed (we are using the symlink Dformer_format)
# config.dataset_path = "datasets/Dformer_format"

C = config
