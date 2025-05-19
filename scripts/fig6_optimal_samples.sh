
BENCHMARK="imagenet1k"
SAVE_DIR="scaling_laws/derived_scaling_laws/fig6"
mkdir -p $SAVE_DIR
LARGE_FLOPS_MAPPING="mammut:1e12,coca:1e10,clip:1e12,siglip:1e10" # Fit the line only up to specified gflops, make predictions beyond that
pareto_method=pareto_naive
max_samples_seen=1e10

python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type line \
    --x_col gflops_total \
    --x_col_pareto gflops_total \
    --y_col total_samples_seen \
    --method curve_fit \
    --models mammut clip \
    --pareto_method $pareto_method \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets datacomp_1b \
    --save_pdf \
    --predict_for_vals 2.14e12 2.59e12 \
    --show_ci \
    --create_figure \
    --max_samples_seen $max_samples_seen \
    --save_latex \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "Optimal Number of Samples \$D_{opt}(C)\$" \
    --min_flops 1e6 \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

# retrival
BENCHMARK="mscoco_captions"
# python -m scaling_laws.main \
#     --benchmark $BENCHMARK \
#     --save_dir $SAVE_DIR \
#     --func_type saturation \
#     --x_col total_samples_seen \
#     --x_col_pareto total_samples_seen \
#     --method curve_fit \
#     --models mammut clip \
#     --pareto_method $pareto_method \
#     --large_flops_mapping $LARGE_FLOPS_MAPPING \
#     --save_extra_cols gflops_total \
#     --pretrain_datasets datacomp_1b \
#     --save_pdf \
#     --create_figure \
#     --save_latex \
#     --show_ci \
#     --min_flops 1e6 \
#     --x_label "Total Samples Seen" \
#     --y_label "image retrieval recall@5 Error Rate" \
#     --max_x 1e10 \
#     --n_bootstraps 1 \
#     --lr_scheduler cosine \
