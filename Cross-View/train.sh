LOGFILE=loggers/${1}.log

CUDA_VISIBLE_DEVICES=2 python3 train_transformer.py > "$LOGFILE" 2>&1 &
# CUDA_VISIBLE_DEVICES=6 python3 train_transformer_cls.py > "$LOGFILE" 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python3 trainClassifier_CV.py > "$LOGFILE" 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python3 train_tenc_recon.py > "$LOGFILE" 2>&1 &