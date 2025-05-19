
BENCHMARK="imagenet-r_test"
SAVE_DIR="scaling_laws/derived_scaling_laws/fig10_extra_evals"
mkdir -p $SAVE_DIR
LARGE_FLOPS_MAPPING="mammut:1e12,coca:1e11,clip:1e12,siglip:1e10" # Fit the line only up to specified gflops, make predictions beyond that
max_x=1e12
pareto_method=pareto_bin
func_type=saturation_both
max_samples_seen=1e10

python -m scaling_laws.main \
    --benchmark $BENCHMARK \
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
    --y_label "ImageNet1k-Robustness [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

BENCHMARK="datacomp_classification_stable_eval"
python -m scaling_laws.main \
    --benchmark $BENCHMARK \
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
    --y_label "DataComp Eval Suite 0-shot [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

BENCHMARK="imagenetv2_test"
python -m scaling_laws.main \
    --benchmark $BENCHMARK \
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
    --y_label "ImageNet1k-V2 [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \
