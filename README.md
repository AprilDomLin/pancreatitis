# Beyond the Pancreas: Diagnosing Pancreatitis and Interpretation Approaches in CT Imaging


## Overview

ViT classification models based on pre-training parameters achieve better diagnostic performance for acute pancreatitis. Using attention-specific interpreters such as the Generic Attention Interpreter (GAI) and Bidirectional Transformer Interpreter (BTI), we analyzed the model's diagnostic basis and unexpectedly found an important contribution of the posterior wall of the body cavity and paraspinal tissues to acute pancreatitis.


## Train & Evaluation

with `.ipynb`

```
# just run main.ipynb in paddle AI studio environment.
```

with `.py`

```
python main.py --model_name ViT_small_patch16_224 --img_size 224
```

# Pancreatitis Dataset
This study receives approval from the Ethics Committee of the People's Hospital of Sichuan Academy of Medical Sciences, and patients with acute pancreatitis (AP) are exempted from providing informed consent.

The PATH of dataset is  `./src/Pancreas.zip`.