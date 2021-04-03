# YueCao Honour Thesis
 
Please check README.pdf for more details

Transformer:
To train the model:
 1.	go to Ran2Ran.py.
 2.	Change the path of dataset in line 62.
 3.	change batch size in line 75.
 4.	change the training parameter in line539-585.
 5.	Run Ran2Ran.PY.
To generate sentence:
 1.	go to b2b_out.py.
 2.	change the path of your dataset in line 61.
 3.	change the path of your checkpoint in line 666.
 4.	Run Ran2Ranoutput.py.

BERT:
To train the model:
 1.	go to BERT2BERT.py.
 2.	change batch size in line 18.
 3.	change the training parameter in line129-144.
 4.	Change the path of dataset in line 151.
 5.	Run BERT2BERT.PY.
To generate sentence:
 1.	go to b2b_out.py.
 2.	change the path of your dataset in line 153.
 3.	change the path of your checkpoint in line 187.
 4.	Run b2b_out.py.

RoBERTa:
To train the model:
 1.	go to yue_Rbmodel.py.
 2.	change batch size in line 18.
 3.	change the training parameter in line129-144.
 4.	Change the path of dataset in line 151.
 5.	Run yue_Rbmodel.py.
To generate sentence:
 1.	go to Rb2Rb_out.py.
 2.	change the path of your dataset in line 156.
 3.	change the path of your checkpoint in line 190.
 4.	Run Rb2Rb_out.py.

RRGen:
To train the model:
 1.	Go to embedding.py and change the path to your data set.
 2.	You can change training parameter in parameter.py. 
 3.	In the parameter.py change pred_test in line 8 to False
 4.	You can change the checkpoint saving path in checkpoint.py.
 5.	Run model.py.
To generate sentence:
 1.	Go to parameter.py and change pred_test in line 8 to True.
 2.	In parameter.py change LOAD_CHECKPOINT in line 7 to True.
 3.	Run model.py.
