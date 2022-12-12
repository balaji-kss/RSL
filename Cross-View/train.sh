LOGFILE=loggers/${1}.log

CUDA_VISIBLE_DEVICES=5 python3 train_transformer.py > "$LOGFILE" 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python3 trainClassifier_CV.py > "$LOGFILE" 2>&1 &

#CUDA_VISIBLE_DEVICES=4 python3 train_tenc_recon.py > "$LOGFILE" 2>&1 &