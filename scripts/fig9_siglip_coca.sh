
BENCHMARK="imagenet1k"
SAVE_DIR="scaling_laws/derived_scaling_laws/fig9_siglip_coca"
mkdir -p $SAVE_DIR

pareto_method=pareto_bin
func_type=saturation_both
max_samples_seen=1e9

LARGE_FLOPS_MAPPING="mammut:1e12,coca:1e11,clip:1e10,siglip:1e10" # Fit the line only up to specified gflops, make predictions beyond that
max_x=1e10

python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip siglip \
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
    --y_label "ImageNet1k 0-shot [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models coca mammut \
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
    --y_label "ImageNet1k 0-shot [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \
