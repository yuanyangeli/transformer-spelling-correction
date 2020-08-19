#export CUDA_VISIBLE_DEVICE=1,2,3
#python train.py -train train.clean.src train.clean.tgt -valid valid.clean.src valid.clean.tgt -vocab vocab.8k.share
python translate.py -input test.clean.src -vocab vocab.8k.share -model_path train-200309-124221
python compare.py
