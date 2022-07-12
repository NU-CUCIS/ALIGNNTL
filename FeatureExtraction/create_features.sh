for filename in ../examples/*.vasp; do
    python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format poscar --file_path "$filename" --output_path "../examples/data"
done
