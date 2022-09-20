LOGFILE=loggers/${1}.log

python3 trainClassifier_CV.py > "$LOGFILE" 2>&1 &