python -u infer.py --data-path /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/010/volume.npy --feature-output-size 128 --slice-along z
python -u infer.py --data-path /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/010/volume.npy --feature-output-size 128 --slice-along y
python -u infer.py --data-path /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/010/volume.npy --feature-output-size 128 --slice-along x
python -u merge_features.py --data /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/010 --name 'volume_vits8_*_features128.npy'
