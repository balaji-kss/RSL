LOGFILE=loggers/${1}.log

python3 transformer_debug1.py > "$LOGFILE" 2>&1 &

