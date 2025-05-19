BENCHMARK="imagenet1k"
SAVE_DIR="scaling_laws/derived_scaling_laws/fig12_less_points_relaion"
mkdir -p $SAVE_DIR
LARGE_FLOPS_MAPPING="mammut:2.5e11,coca:1e11,clip:2.5e11,siglip:1e10" # Fit the line only up to specified gflops, make predictions beyond that
max_x=2e12
pareto_method=pareto_bin
func_type=saturation
max_samples_seen=1e10
dataset="datacomp_1b"

python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets $dataset \
    --save_pdf \
    --pareto_method $pareto_method \
    --create_figure \
    --save_latex \
    --min_flops 1e6 \
    --max_x $max_x \
    --save_extra_cols model total_samples_seen \
    --max_samples_seen $max_samples_seen \
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "ImageNet1k 0-shot [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

# # retrieval 
python -m scaling_laws.main \
    --benchmark mscoco_captions \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets $dataset \
    --save_pdf \
    --pareto_method $pareto_method \
    --create_figure \
    --save_latex \
    --min_flops 1e6 \
    --save_extra_cols model total_samples_seen \
    --max_x $max_x \
    --max_samples_seen $max_samples_seen \
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "MSCOCO Image Retrival R@5 [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

LARGE_FLOPS_MAPPING="mammut:5e11,coca:1e11,clip:5e11,siglip:1e10"
python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets $dataset \
    --save_pdf \
    --pareto_method $pareto_method \
    --create_figure \
    --save_latex \
    --min_flops 1e6 \
    --save_extra_cols model total_samples_seen \
    --max_samples_seen $max_samples_seen \
    --show_ci \
    --max_x $max_x \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "ImageNet1k 0-shot [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

# retrieval 
python -m scaling_laws.main \
    --benchmark mscoco_captions \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets datacomp_1b \
    --save_pdf \
    --pareto_method $pareto_method \
    --create_figure \
    --save_latex \
    --min_flops 1e6 \
    --save_extra_cols model total_samples_seen \
    --max_x $max_x \
    --max_samples_seen $max_samples_seen \
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "MSCOCO Image Retrival R@5 [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \