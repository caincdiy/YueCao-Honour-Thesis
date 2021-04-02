import torch
import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/ex-fhendija-1/caoyuecc/Rbcache/'
from torch.utils.data import DataLoader
import transformers as transf
from transformers import AdamW, Trainer, TrainingArguments

from transformers import RobertaTokenizer, RobertaModel, EncoderDecoderModel

import numpy as np
import pandas as pd

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

PER_TRAINED_MODEL_NAME="roberta-base"
MAX_LEN=256
BATCH_SIZE=32

tokenizer = RobertaTokenizer.from_pretrained(PER_TRAINED_MODEL_NAME)

"""
This part of code was adapted from a post from V. Valkov
- Title: Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python
- Author: V. Valkov
- Access date: Jan.10 2021
- Availability: https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
"""

class RRDataset ():
  def __init__(self, reviews, reply, tokenizer, max_len):
    self.reviews=reviews
    self.reply=reply
    self.tokenizer=tokenizer
    self.max_len=max_len

  def __len__ (self):
    return len(self.reviews)
  
  def __getitem__ (self, item):
    review= str(self.reviews[item])
    reply= str(self.reply[item])
    
    tokenized_review = self.tokenizer(
      review,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    ).to('cuda')

    tokenized_reply = self.tokenizer(
      reply,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    ).to('cuda')

    labels=self.tokenizer(
      reply,
      max_length=self.max_len,
      padding='max_length',
      truncation=True,
    ).input_ids
    labels=[ -100 if token==tokenizer.pad_token_id else token for token in labels]
    labels=torch.tensor(labels).to('cuda')
    return{
         'input_ids':tokenized_review['input_ids'].flatten(),
         'attention_mask': tokenized_review['attention_mask'].flatten(),
         'decoder_input_ids': tokenized_reply['input_ids'].flatten(),
         'decoder_attention_mask':tokenized_reply['attention_mask'].flatten(),
         'labels':labels

     }




def translate_sentence(src,model):
  inputs = tokenizer(src, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
  input_ids = inputs.input_ids.to("cuda")
  attention_mask = inputs.attention_mask.to("cuda")
  outputs=model.generate(input_ids, attention_mask=attention_mask)
  output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  return output_str

def calculate_bleu(data, model):
    count=0
    trgs = []
    srcs=[]
    pred_trgs = []
    for i in range(len(data)): #len(data)
        #if count==10:
          #break
        pr = ""
        tr = ""
        sr = ""
        src = data['review'][i]
        trg = data['reply'][i]
        
        pred_trg= translate_sentence(src,model)
        
       
        pred_trgs.append(pred_trg[0])
        trgs.append(trg)
        srcs.append(src)
        count+=1

    with open("preds.txt", "w", encoding="utf-8") as f:
      for i in pred_trgs:
        f.write(i+"\n")
    with open("given.txt", "w", encoding="utf-8") as f:
      for i in trgs:
        f.write(i+"\n")
    outcsv={'review':srcs,'predicted reply':pred_trgs,'trg reply':trgs}
    outcsvdf=pd.DataFrame(outcsv)
    outcsvdf.to_csv("Whole_Review_Reply.csv",index=False)

def main():



    training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=5,
    fp16=True, 
    output_dir="checkpoints/Batch32",
    logging_steps=2,
    save_steps=5985,
    eval_steps=5985,
    warmup_steps=2000
    # logging_steps=1000,
    # save_steps=500,
    # eval_steps=7500,
    # warmup_steps=2000,
    # save_total_limit=3,
    )

    

    
    
    train_data=pd.read_csv("/scratch/ex-fhendija-1/caoyuecc/Data/train_data.csv") 
    test_data=pd.read_csv("/scratch/ex-fhendija-1/caoyuecc/Data/test_data.csv")
    val_data=pd.read_csv("/scratch/ex-fhendija-1/caoyuecc/Data/val_data.csv")
    
    
    train_dataset=RRDataset(
      reviews=train_data.review.to_numpy(),
      reply=train_data.reply.to_numpy(),
      tokenizer=tokenizer,
      max_len=MAX_LEN
                    )

    test_dataset=RRDataset(
          reviews=test_data.review.to_numpy(),
          reply=test_data.reply.to_numpy(),
          tokenizer=tokenizer,
          max_len=MAX_LEN
                        )

    val_dataset=RRDataset(
          reviews=val_data.review.to_numpy(),
          reply=val_data.reply.to_numpy(),
          tokenizer=tokenizer,
          max_len=MAX_LEN
                        )
                        
                        
    #tokenizer.convert_ids_to_tokens(test_token['input_ids'][0])

    #print(train_dataset[0])


    #model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    model=EncoderDecoderModel.from_encoder_decoder_pretrained(PER_TRAINED_MODEL_NAME,PER_TRAINED_MODEL_NAME)


    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size=model.config.encoder.vocab_size

    model.config.max_length = 256
    model.config.min_length = 70

    model.config.no_repeat_ngram_size = 2
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    model.to('cuda')
    
    
    
    
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    


    
    
if __name__ == '__main__':
  main()