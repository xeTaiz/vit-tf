find /mnt/hdd/dome/ntf_volumes -name 'volume.npy' -exec python infer.py --data-path {} --feature-output-size 96 --slice-along all \;
