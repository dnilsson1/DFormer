#!/bin/bash
cd /workspace
python -c "
from local_configs.JARVIS.DFormer_Large_custom import C
print('Config imported successfully')
print('Dataset:', C.dataset_name)
print('Backbone:', C.backbone)
print('Classes:', C.num_classes)
print('Batch size:', C.batch_size)
print('Learning rate:', C.lr)
"