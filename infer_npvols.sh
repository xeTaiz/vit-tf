find /mnt/hdd/dome/ntf_volumes -name '*.npy' -exec python infer.py --data-path {} --feature-output-size 128 --slice-along all \;
find /mnt/hdd/dome/ntf_volumes -name '*.npy' -exec python infer.py --data-path {} --feature-output-size 128 --slice-along z \;
find /mnt/hdd/dome/ntf_volumes -name '*.npy' -exec python infer.py --data-path {} --feature-output-size 128 --slice-along all --dino2-model vits14 \;
