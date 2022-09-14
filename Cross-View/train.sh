LOGFILE=logs/${1}.log

python3 trainClassifier_Multi_nobc.py > "$LOGFILE" 2>&1 &
#python3 train_classifier.py > "$LOGFILE" 2>&1 &
#python3 train_cls_dyn.py > "$LOGFILE" 2>&1 &