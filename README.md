# E2E_ASD_DETECTION

ASD Detection based on feature extractors(Auto-encoder or wav2vec2.0) and BLSTM classifier

# Information
An end-to-end model training and evaluation scripts based on wav2vec2.0 model and BLSTM classifier. Extract vocal characteristics with wav2vec2.0 from children's voice segments and classify it with BLSTM classifier as ASD/TD. <br />

# Arguments
main.py [--train] [--eval] [--target_model ft, jt] [--exp exp]

# Usage
1. Set wave files and corresponding paths in data/*.csv.
2. Train model with ```--train``` command. You can select fine-tuning based model or joint-optimization by ```--target_model``` argument, and experiment name with ```--exp``` argument.
```
python main.py --train --target_model ft --exp ft_test
```
3. After the training is done, you can evaluate your model with ```--eval``` argument with ```--target_model``` and ```--exp``` given in 2.
```
python main.py --eval --target_model ft --exp ft_test
```

# Acknowledgement
This work was supported by the Institute of Information & communications Technology Planning & evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2019-0-00330, Development of AI Technology for Early Screening of Child/Child Autism Spectrum Disorders based on Cognition of the Psychological Behavior and Response).
