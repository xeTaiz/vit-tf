find /mnt/hdd/dome/ntf_volumes/CT-ORG/npy -name 'volume.npy' -exec python infer.py --data-path {} --feature-output-size 64 --slice-along all \;
