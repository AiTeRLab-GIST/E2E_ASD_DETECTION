# E2E_ASD_DETECTION

ASD Detection based on feature extractors(Auto-encoder or wav2vec2.0) and BLSTM classifier

# Information
The model training and evaluation scripts for BLSTM, AE-BLSTM-FT, AE-BLSTM-JT, W2V-BLSTM-FT, and W2V-BLSTM-JT. <br />

# Arguments
For AE-BLSTM-FT, it consists of two stages. You should train the model rgrs and clsf in sequence.
```
python main.py [--train] [--eval] [--target_model rgrs, clsf] [--exp exp]
```
All the other models only uses following command:
```
python main.py [--train] [--eval] [--exp exp]
```

# Usage
1. Set wave files and corresponding paths in data/*.csv.
2. Train model with ```--train``` command. You can select experiment name with ```--exp``` argument.
```
python main.py --train --exp ft_test
```
3. After the training is done, you can evaluate your model with ```--eval``` argument ```--exp``` given in 2.
```
python main.py --eval --exp ft_test
```

# Acknowledgement
This work was supported by the Institute of Information & communications Technology Planning & evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2019-0-00330, Development of AI Technology for Early Screening of Child/Child Autism Spectrum Disorders based on Cognition of the Psychological Behavior and Response).
