LOGFILE=loggers/${1}.log

CUDA_VISIBLE_DEVICES=1 python3 transformer_debug1.py > "$LOGFILE" 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python3 trainClassifier_CV.py > "$LOGFILE" 2>&1 &

#CUDA_VISIBLE_DEVICES=0 python3 train_tenc_recon.py > "$LOGFILE" 2>&1 &