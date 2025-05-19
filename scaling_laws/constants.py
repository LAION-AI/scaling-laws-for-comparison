import seaborn as sns

metric_map = {
    "imagenet1k": "acc1",
    "mscoco_captions": "image_retrieval_recall@5",
    "winoground": "acc",
    "relaion2b": "contrastive_loss",
    "datacomp_classification": "acc1",
    "ade20k": "miou",
    "imagenet-r_test": "acc1",
    "imagenetv2_test": "acc1",
    "imagenet_distribution_shift": "acc1",
    "datacomp_classification_stable_eval": "acc1",
}


sns_colors = sns.color_palette("tab10")

colors = {
    "mammut": sns_colors[0],
    "coca": sns_colors[1],
    "clip": sns_colors[2],
    "siglip": sns_colors[6],
    "cap": sns_colors[7],
}

dataset_colors = {
    "LAION-2B": sns_colors[0],
    "LAION-400M": sns_colors[1],
    "LAION-80M": sns_colors[2],
    "datacomp_1b": sns_colors[3],
    "relaion2b-en": sns_colors[4],
    "CLIP-WIT": sns_colors[5],
}

markers = {
    "LAION-2B": "o",
    "LAION-400M": "s",
    "LAION-80M": "D",
    "datacomp_1b": "P",
}
