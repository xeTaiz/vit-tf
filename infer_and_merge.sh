python infer.py --data-path /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/010/volume.npy --feature-output-size 128 --slice-along z 
python infer.py --data-path /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/010/volume.npy --feature-output-size 128 --slice-along y 
python infer.py --data-path /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/010/volume.npy --feature-output-size 128 --slice-along x 
python merge_features.py --data /mnt/hdd/dome/ntf_volumes/CT-ORG/npy/010 --name 'volume_vits8_*_features128.npy'
