find /mnt/hdd/dome/ntf_volumes -name 'volume.npy' -not -path "/mnt/hdd/dome/ntf_volumes/CT-ORG/*" -exec python infer.py --data-path {} --feature-output-size 128 --slice-along all \;
find /mnt/hdd/dome/ntf_volumes -name 'volume.npy' -not -path "/mnt/hdd/dome/ntf_volumes/CT-ORG/*" -exec python infer.py --data-path {} --feature-output-size 96 --slice-along all \;
