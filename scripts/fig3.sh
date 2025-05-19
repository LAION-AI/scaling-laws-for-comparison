
BENCHMARK="imagenet1k"
SAVE_DIR="scaling_laws/derived_scaling_laws/fig3"
mkdir -p $SAVE_DIR
LARGE_FLOPS_MAPPING="mammut:1.3e10,coca:1e10,clip:1.3e10" # Fit the line only up to specified gflops, make predictions beyond that
pareto_method=pareto_naive
max_samples_seen=1.3e10
func_type=saturation_both


python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col total_samples_seen \
    --x_col_pareto total_samples_seen \
    --method curve_fit \
    --models mammut clip \
    --pareto_method $pareto_method \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --save_extra_cols gflops_total \
    --pretrain_datasets datacomp_1b \
    --save_pdf \
    --predict_for_vals  2.3e10 2.64e10 1.23e10 1.42e10 \
    --max_samples_seen $max_samples_seen \
    --create_figure \
    --save_latex \
    --show_ci \
    --x_label "Number of Samples" \
    --y_label "ImageNet1k 0-shot [Error Rate]" \
    --min_flops 1e6 \
    --max_x 1.3e10 \
    --n_bootstraps 1 \
    --lr_scheduler cosine \


# retrival
BENCHMARK="mscoco_captions"
python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col total_samples_seen \
    --x_col_pareto total_samples_seen \
    --method curve_fit \
    --models mammut clip \
    --pareto_method $pareto_method \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --save_extra_cols gflops_total \
    --pretrain_datasets datacomp_1b \
    --save_pdf \
    --predict_for_vals 2.14e12 2.59e12 \
    --create_figure \
    --save_latex \
    --show_ci \
    --max_samples_seen $max_samples_seen \
    --min_flops 1e6 \
    --x_label "Number of Samples" \
    --y_label "MSCOCO image retrieval R@5 [Error Rate]" \
    --max_x 1.3e10 \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

