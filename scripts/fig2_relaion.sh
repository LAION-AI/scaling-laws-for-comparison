
BENCHMARK="imagenet1k"
SAVE_DIR="scaling_laws/derived_scaling_laws/fig2_relaion"
mkdir -p $SAVE_DIR
LARGE_FLOPS_MAPPING="mammut:1e12,coca:1e10,clip:1e12" # Fit the line only up to specified gflops, make predictions beyond that
DATASET=relaion2b-en
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
    --pretrain_datasets $DATASET \
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

max_x=1e12
# retrieval 
python -m scaling_laws.main \
    --benchmark mscoco_captions \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets $DATASET \
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


# fitting with less data
LARGE_FLOPS_MAPPING="mammut:1e11,coca:1e10,clip:1e11"
python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets $DATASET \
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

# retrieval
python -m scaling_laws.main \
    --benchmark mscoco_captions \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets $DATASET \
    --save_pdf \
    --pareto_method $pareto_method \
    --create_figure \
    --save_latex \
    --save_extra_cols model total_samples_seen \
    --max_x $max_x \
    --max_samples_seen $max_samples_seen \
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "MSCOCO Image Retrival R@5 [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

# one more point
LARGE_FLOPS_MAPPING="mammut:2e11,coca:1e10,clip:2e11"
python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets $DATASET \
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

# retrieval
python -m scaling_laws.main \
    --benchmark mscoco_captions \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets $DATASET \
    --save_pdf \
    --pareto_method $pareto_method \
    --create_figure \
    --save_latex \
    --save_extra_cols model total_samples_seen \
    --max_x $max_x \
    --max_samples_seen $max_samples_seen \
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "MSCOCO Image Retrival R@5 [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \
