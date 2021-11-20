
VERSION=v11
LOGFILE=logs/exp_${VERSION}.log

python3 train.py > "$LOGFILE" 2>&1 &