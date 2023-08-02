find /mnt/hdd/dome/ntf_volumes -name '*.npy' -exec python infer.py --data-path {} --feature-output-size 128 \;
