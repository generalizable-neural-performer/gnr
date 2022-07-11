genebody_path=$1

subjects=($(ls ${genebody_path}))
for ((i=0; i<${#genebody_path[@]}; ++i)) do
    subject=${subjects[i]}
    if [[ -d "${genebody_path}/${subject}" ]]
    then
        python apps/render_smpl_depth.py --datadir ${genebody_path}/${subject} 
    fi
done