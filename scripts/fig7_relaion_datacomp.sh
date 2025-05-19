
BENCHMARK="imagenet1k"
SAVE_DIR="scaling_laws/derived_scaling_laws/fig7_relaion_datacomp"
mkdir -p $SAVE_DIR
LARGE_FLOPS_MAPPING="mammut:1e13,coca:1e10,clip:1e13" # Fit the line only up to specified gflops, make predictions beyond that
max_x=1e13
DATASET=relaion2b-en
func_type=saturation_both
pareto_method=pareto_bin
max_samples_seen=1e13

python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip  \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets datacomp_1b relaion2b-en CLIP-WIT \
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

max_samples_seen=1e10
LARGE_FLOPS_MAPPING="mammut:1e12,coca:1e10,clip:1e12" # Fit the line only up to specified gflops, make predictions beyond that
python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets relaion2b-en datacomp_1b \
    --save_pdf \
    --pareto_method pareto_bin \
    --create_figure \
    --max_samples_seen $max_samples_seen \
    --save_latex \
    --min_flops 1e6 \
    --save_extra_cols model total_samples_seen \
    --max_x $max_x \
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "ImageNet1k 0-shot [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

retrieval 
BENCHMARK="mscoco_captions"
LARGE_FLOPS_MAPPING="mammut:1e13,coca:1e10,clip:1e13" # Fit the line only up to specified gflops, make predictions beyond that
max_x=1e13
max_samples_seen=1e13
python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models clip \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets relaion2b-en datacomp_1b CLIP-WIT \
    --save_pdf \
    --pareto_method pareto_bin \
    --create_figure \
    --save_latex \
    --min_flops 1e6 \
    --save_extra_cols model total_samples_seen \
    --max_x $max_x \
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "MSCOCO Image Retrival R@5 [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

max_samples_seen=1e10
LARGE_FLOPS_MAPPING="mammut:1e12,coca:1e10,clip:1e12" # Fit the line only up to specified gflops, make predictions beyond that
python -m scaling_laws.main \
    --benchmark $BENCHMARK \
    --save_dir $SAVE_DIR \
    --func_type $func_type \
    --x_col gflops_total \
    --method curve_fit \
    --models mammut \
    --large_flops_mapping $LARGE_FLOPS_MAPPING \
    --pretrain_datasets relaion2b-en datacomp_1b CLIP-WIT \
    --save_pdf \
    --pareto_method pareto_bin \
    --create_figure \
    --save_latex \
    --min_flops 1e6 \
    --save_extra_cols model total_samples_seen \
    --max_x $max_x \
    --show_ci \
    --x_label "Compute \$C\$ [GFLOPs]" \
    --y_label "MSCOCO Image Retrival R@5 [Error Rate]" \
    --n_bootstraps 1 \
    --lr_scheduler cosine \

