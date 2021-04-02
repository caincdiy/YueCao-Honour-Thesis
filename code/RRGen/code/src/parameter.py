from embedding import AttrDict
from checkpoint import load_checkpoint
import os


# If enabled, load checkpoint.
LOAD_CHECKPOINT = True
pred_test = True # Predict the test data

if LOAD_CHECKPOINT:
    # Modify this path.
    #def get_path(i,j):
    #    out_dir = "/scratch/ex-fhendija-1/caoyuecc/RRGen/RRGenCode/src/checkpoints/accuracy84.28191997884242_step16800_checkpoint.pt"
    #    checkpoint_dirs = os.listdir(out_dir)
    #    # for idx, checkpoint_dir in enumerate(checkpoint_dirs):
    #    #     checkpoint_fns = os.listdir(os.path.join(out_dir, checkpoint_dir))
    #    checkpoint_fns = os.listdir(os.path.join(out_dir, checkpoint_dirs[i]))
    #    # for jdx, checkpoint_fn in enumerate(checkpoint_fns[i]):
    #    checkpoint_path = os.path.join(out_dir, checkpoint_dirs[i], checkpoint_fns[j])
    #    print("Current checkpoint path is ", checkpoint_path)
    #    return checkpoint_path
    #checkpoint_path = get_path(7,17)
    checkpoint = load_checkpoint("/scratch/ex-fhendija-1/caoyuecc/RRGen/RRGenCode/src/checkpoints/1o3/accuracy67.28466549517381_step49875_checkpoint.pt")
    opts = checkpoint['opts']
    print('=' * 100)
    print('Options log:')
    print('- Load from checkpoint: {}'.format(LOAD_CHECKPOINT))
    print('- Global step: {}'.format(checkpoint['global_step']))

else:
    opts = AttrDict()
    # Configure models
    opts.word_vec_size = 100
    opts.feature_vec_size = 90
    opts.rnn_type = 'GRU'
    opts.hidden_size = 200
    opts.batch_size = 32
    opts.max_vocab_size = 10000
    opts.num_layers = 1
    opts.dropout = 0.1
    opts.bidirectional = True
    opts.attention = True
    opts.share_embeddings = True
    opts.pretrained_embeddings = True
    opts.fixed_embeddings = False
    opts.tie_embeddings = True  # Tie decoder's input and output embeddings
    # opts.pred_test = True  # Predict the test data

    # Configure external features
    opts.use_sent_rate = True
    opts.use_sent_senti = True   # consider review sentiment
    opts.use_sent_len = 20  # consider review length 20, the categorization interval, else "False"
    opts.use_app_cate = True    # consider app category
    opts.use_keyword = True
    opts.tie_ext_feature = True   # Tie external feature embeddings


    # Configure optimization
    opts.max_grad_norm = 2
    opts.learning_rate = 0.001
    opts.weight_decay = 1e-5  # L2 weight regularization


    # Configure training
    opts.max_seq_len = 256  # max sequence length to prevent OOM.
    opts.num_epochs = 25
    opts.print_every_step = 1995
    opts.save_every_step = 1995
    for k, v in opts.items(): print('- {}: {}'.format(k, v))
    print('=' * 100 + '\n')

    # Configure attention result file
    # opts.attention_res_fp = '/research/lyu1/jczeng/cygao/attention'
    # opts.outtext_fp = '/research/lyu1/jczeng/cygao/texts'
    opts.attention_res_fp = '/workspace/results/attention'
    opts.outtext_fp = '/workspace/results/texts'

