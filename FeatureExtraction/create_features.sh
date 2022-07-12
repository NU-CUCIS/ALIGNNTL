for filename in /data/vgf3011/alignntldata/jid/*.vasp; do
    python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format poscar --file_path "$filename" --output_path "/data/vgf3011/alignntldata/data"
done
