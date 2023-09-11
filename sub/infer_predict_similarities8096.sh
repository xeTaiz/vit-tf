while read i; do python -u predict_ntf.py --data /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/$i --bilateral-solver --num-samples 8096; done <volumes_for_metrics.txt
