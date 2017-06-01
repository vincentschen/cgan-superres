# Overview
- Conditional gan for super-resolution.
- Using jupyter notebooks for development. 

# Setup 

## Environment 
1. `virtualenv -p python3 .env`
1. `pip install -r requirements.txt`
1. Do this in the *ROOT* folder to avoid committing outputs of jupyter-notebook: https://github.com/jond3k/ipynb_stripout
1. (optional) Setup save hooks to automatically convert ipython to .py: http://jupyter-notebook.readthedocs.io/en/latest/extending/savehooks.html


## Format Data 
1. Make sure file structure is the following (from current working directory): 

  <pre>
  .
  +-- datasets/
  |   +-- celeba/
  |   |   +-- aligned_cropped/
  |   |   +-- Anno/
  |   |   +-- list_eval_partition.txt 
  </pre>

1. Then, run the following formatting script to convert raw data and attributes into `.tfrecord` file:
```
python celeba_formatting.py \
    --partition_fn ./datasets/celeba/list_eval_partition.txt \
    --file_out ./datasets/celeba/celeba_train \
    --fn_root ./datasets/celeba/aligned_cropped \
    --set 0
```

  - Options 
    - --set 0 # train 
    - --set 1 # val
    - --set 2 # test