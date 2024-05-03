import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out' #'out-funcom_raw_scratch_1b_local'
eval_interval = 100
eval_iters = 100
wandb_log = False # feel free to turn on
wandb_project = 'funcom_raw'
wandb_run_name = 'ft-jam350m-soj-1' #+ str(time.time())

dataset = 'eldenRingModel'
#init_from = 'scratch'
#init_from = 'gpt2-medium'
#init_from = 'gpt2-large'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = True

#n_layer = 12
#n_head = 6
#n_embd = 768
#dropout = 0.2

block_size = 1024

# gpt2-large
#n_layer = 36
#n_head = 20
#n_embd = 1280
#dropout = 0.2

# gpt2-medium
#n_layer = 24
#n_head = 16
#n_embd = 1024
#dropout = 0.2

#n_layer = 32
#n_head = 32
#n_embd = 1536
#dropout = 0.2

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters

# stackoverflow has 10,495,518,108 tokens
# openwebtext has 9,035,582,489 tokens
# funcom_raw has 8,752,695,577 tokens

# eldenRingModel has 17144353 tokens
# 2nd training set has 34293545 tokens
batch_size = 2 #16
gradient_accumulation_steps = 32
#max_iters = 525 #iterations from train from tmp
max_iters = 1050 + 2100

#learning_rate = 3e-5
weight_decay = 1e-1
#decay_lr = False

