
BENCHMARK="imagenet1k"
SAVE_DIR="scaling_laws/derived_scaling_laws/fig4_const"
mkdir -p $SAVE_DIR
LARGE_FLOPS_MAPPING="mammut:5e11,coca:1e10,clip:5e11" # Fit the line only up to specified gflops, make predictions beyond that
max_x=2e11
pareto_method=pareto_bin
lr_scheduler=const
func_type=saturation_both

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
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "ImageNet1k 0-shot [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler $lr_scheduler \


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
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "MSCOCO Image Retrival R@5 [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler $lr_scheduler \

# one more point
# LARGE_FLOPS_MAPPING="mammut:2e11,coca:1e10,clip:2e11"
# python -m scaling_laws.main \
#     --benchmark $BENCHMARK \
#     --save_dir $SAVE_DIR \
#     --func_type saturation \
#     --x_col gflops_total \
#     --method curve_fit \
#     --models clip mammut \
#     --large_flops_mapping $LARGE_FLOPS_MAPPING \
#     --pretrain_datasets datacomp_1b \
#     --save_pdf \
#     --pareto_method $pareto_method \
#     --create_figure \
#     --save_latex \
#     --min_flops 1e6 \
#     --save_extra_cols model total_samples_seen \
#     --max_x $max_x \
#     --show_ci \
#     --x_label GFLOPS \
#     --y_label "image retrieval recall@5 Error Rate" \
#     --n_bootstraps 1 \
#     --lr_scheduler $lr_scheduler \

# # retrieval
# python -m scaling_laws.main \
#     --benchmark mscoco_captions \
#     --save_dir $SAVE_DIR \
#     --func_type saturation \
#     --x_col gflops_total \
#     --method curve_fit \
#     --models clip mammut \
#     --large_flops_mapping $LARGE_FLOPS_MAPPING \
#     --pretrain_datasets datacomp_1b \
#     --save_pdf \
#     --pareto_method $pareto_method \
#     --create_figure \
#     --save_latex \
#     --save_extra_cols model total_samples_seen \
#     --max_x $max_x \
#     --show_ci \
#     --x_label GFLOPS \
#     --y_label "image retrieval recall@5 Error Rate" \
#     --n_bootstraps 1 \
#     --lr_scheduler $lr_scheduler \

