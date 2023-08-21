while read i; do python -u predict_similarities_ntf.py --data /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/$i --bilateral-solver --num-samples 1024; done <volumes_for_metrics.txt
