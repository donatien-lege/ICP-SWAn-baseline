configfile: "../config/params.yml"
folds = range(1, config["n_folds"] +1)

rule all:
    input:
        expand("../reports/metrics/{model}/set_{s}.csv",
            model=config["models"], s=config["sets"],
        ),
        expand("../reports/best_params/{model}/clf.csv", model=config["models"]),
        expand("../reports/best_params/{model}/reg.csv", model=config["models"]),
        expand("../reports/naive_metrics/{model}.csv", model=config["models"]),

#indices for cross-validation + calibration
rule split:
    params:
        folder="../data/train"
    output:
        spl="../results/spl.csv"
    script:
        "scripts/split.py"

#data augmentation
rule augmentation:
    params:
        folder="../data/train"
    input:
        spl="../results/spl.csv"
    output:
        pulses="../results/augmentation/pulses/{fold}.csv",
        labels="../results/augmentation/labels/{fold}.csv"
    script:
        "scripts/augmentation.py"

#training classifier
rule cv_clf:
    input:
        pulses=expand("../results/augmentation/pulses/{fold}.csv", fold=folds),
        labels=expand("../results/augmentation/labels/{fold}.csv", fold=folds), 
    output:
        prec="../results/cv_clf/{model}/{fold}.npy",
    script:
        "scripts/cv_clf.py"

rule train_clf:
    input:
        precs=expand("../results/cv_clf/{{model}}/{fold}.npy", fold=folds),
        pulses=expand("../results/augmentation/pulses/{fold}.csv", fold=folds),
        labels=expand("../results/augmentation/labels/{fold}.csv", fold=folds), 
    output:
        clf="../results/train_clf/{model}/clf.gz",
        params_clf="../reports/best_params/{model}/clf.csv"
    script:
        "scripts/train_clf.py"

#training regressor
rule cv_reg:
    input:
        pulses=expand("../results/augmentation/pulses/{fold}.csv", fold=folds),
        labels=expand("../results/augmentation/labels/{fold}.csv", fold=folds), 
    output:
        score="../results/cv_reg/{model}/{fold}.npy",
    script:
        "scripts/cv_reg.py"

rule train_reg:
    input:
        scores=expand("../results/cv_reg/{{model}}/{fold}.npy", fold=folds),
        pulses=expand("../results/augmentation/pulses/{fold}.csv", fold=folds),
        labels=expand("../results/augmentation/labels/{fold}.csv", fold=folds), 
    output:
        reg_P1="../results/train_reg/{model}/regP1.gz",
        reg_P2="../results/train_reg/{model}/regP2.gz",
        params_reg="../reports/best_params/{model}/reg.csv"
    script:
        "scripts/train_reg.py"

#calibration
rule calibration:
    input:
        spl="../results/spl.csv",
        clf="../results/train_clf/{model}/clf.gz",
        reg_P1="../results/train_reg/{model}/regP1.gz",
        reg_P2="../results/train_reg/{model}/regP2.gz",
    output:
        thresholds="../results/calibration/{model}/set_{s}.csv"
    script:
        "scripts/calibration.py"

#compute naive thresholds based on a ROC curve
#the calibration set is used here as a classical validation set

rule naive_thresholds:
    input:
        spl="../results/spl.csv",
        clf="../results/train_clf/{model}/clf.gz",
    output:
        thresholds="../results/naive_calibration/{model}.csv"
    script:
        "scripts/naive_thresholds.py"

#test
rule eval_model:
    params:
        folder="../data/wei2024",
        sampling_rate=400,
    input:
        thresholds="../results/calibration/{model}/set_{s}.csv",
        clf="../results/train_clf/{model}/clf.gz",
        reg_P1="../results/train_reg/{model}/regP1.gz",
        reg_P2="../results/train_reg/{model}/regP2.gz",
    output:
        "../reports/metrics/{model}/set_{s}.csv"
    script:
        "scripts/eval_model.py"

#test with naive threshold
rule eval_naive_model:
    params:
        folder="../data/wei2024",
        sampling_rate=400,
    input:
        thresholds="../results/naive_calibration/{model}.csv",
        clf="../results/train_clf/{model}/clf.gz",
        reg_P1="../results/train_reg/{model}/regP1.gz",
        reg_P2="../results/train_reg/{model}/regP2.gz",
    output:
        "../reports/naive_metrics/{model}.csv"
    script:
        "scripts/eval_model.py"