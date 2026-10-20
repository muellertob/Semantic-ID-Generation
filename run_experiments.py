#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import shutil
import time
import math
from datetime import datetime
from omegaconf import OmegaConf

# helper to load, update using OmegaConf, and save a fully self-contained configuration file
def prepare_run_config(config_path, dest_config_path, overrides_dict):
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    for k, v in overrides_dict.items():
        OmegaConf.update(config, k, v, force_add=True)
    os.makedirs(os.path.dirname(dest_config_path), exist_ok=True)
    OmegaConf.save(config, dest_config_path)
    return dest_config_path

# prepares config and returns command args, model/output paths for a quantizer training task
def prepare_quantizer_task(exp, seed, dry_run):
    exp_name = exp["name"]
    semids_output_path = os.path.join(OUTPUTS_DIR, f"{exp_name.lower()}_seed_{seed}", "semids.pt")
    quant_meta_json = os.path.join(EXPERIMENT_DIR, f"{exp_name.lower()}_seed_{seed}_quantizer_meta.json")
    quant_run_config = os.path.join(CONFIGS_DIR, f"{exp_name}_seed_{seed}_quantizer.yaml")
    
    if exp.get("is_rqkmeans"):
        overrides = {
            "general.seed": seed,
            "general.output_path": semids_output_path,
            "general.model_path": os.path.join(MODELS_DIR, "rqkmeans", f"{exp_name.lower()}_seed_{seed}.pt"),
            "general.meta_json": quant_meta_json,
            "general.wandb_dir": os.path.join(OUTPUTS_DIR, "wandb", "runs"),
        }
        if dry_run:
            overrides["model.n_iters"] = 2
        prepare_run_config(exp["quant_config"], quant_run_config, overrides)
        cmd = ["python", "main.py", "train-rqkmeans", "--config", quant_run_config]
        model_path = overrides["general.model_path"]
    else:
        overrides = {
            "general.seed": seed,
            "general.model_path": os.path.join(MODELS_DIR, "quantizer", f"{exp_name.lower()}_seed_{seed}.pt"),
            "general.meta_json": quant_meta_json,
            "general.wandb_dir": os.path.join(OUTPUTS_DIR, "wandb", "runs"),
        }
        if dry_run:
            overrides["train.num_epochs"] = 1
            overrides["general.use_wandb"] = False
        prepare_run_config(exp["quant_config"], quant_run_config, overrides)
        cmd = ["python", "main.py", "train-quantizer", "--config", quant_run_config]
        model_path = overrides["general.model_path"]
        
    return cmd, model_path, semids_output_path, quant_meta_json

# runs training and testing for a Seq2Seq recommender and returns metrics and run name
def run_recommender_task(exp, seed, semids_output_path, dry_run):
    exp_name = exp["name"]
    train_meta_json = os.path.join(EXPERIMENT_DIR, f"{exp_name.lower()}_seed_{seed}_train_meta.json")
    test_meta_json = os.path.join(EXPERIMENT_DIR, f"{exp_name.lower()}_seed_{seed}_test_meta.json")
    
    # Train
    rec_overrides = {
        "general.seed": seed,
        "general.meta_json": train_meta_json,
        "general.recommender_model_dir": os.path.join(MODELS_DIR, "recommender"),
        "general.wandb_dir": os.path.join(OUTPUTS_DIR, "wandb", "runs"),
    }
    if dry_run:
        rec_overrides["seq2seq.num_epochs"] = 1
        rec_overrides["seq2seq.eval_epoch_interval"] = 1
        rec_overrides["seq2seq.eval_delay_epochs"] = 0
        rec_overrides["seq2seq.early_stopping"] = False
        rec_overrides["general.use_wandb"] = False
        
    rec_run_config = os.path.join(CONFIGS_DIR, f"{exp_name}_seed_{seed}_recommender.yaml")
    prepare_run_config(exp["recommender_config"], rec_run_config, rec_overrides)
    
    train_cmd = ["python", "main.py", "train-seq2seq", "--config", rec_run_config, "--semids", semids_output_path]
    run_command(train_cmd)
    
    recommender_run_name = None
    best_model_path = None
    if os.path.exists(train_meta_json):
        with open(train_meta_json, "r") as f:
            meta = json.load(f)
        recommender_run_name = meta.get("recommender_run_name")
        best_model_path = meta.get("best_model_path")
        
    if not recommender_run_name:
        recommender_run_name = f"{exp_name}_seed_{seed}"
    if not best_model_path:
        best_model_path = os.path.join(MODELS_DIR, "recommender", f"{recommender_run_name}_best.pt")
        
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Expected recommender model checkpoint not found at: {best_model_path}")
        
    # Test
    test_overrides = {
        "general.meta_json": test_meta_json
    }
    test_run_config = os.path.join(CONFIGS_DIR, f"{exp_name}_seed_{seed}_recommender_test.yaml")
    prepare_run_config(rec_run_config, test_run_config, test_overrides)
    
    test_cmd = [
        "python", "main.py", "test-seq2seq",
        "--config", test_run_config,
        "--semids", semids_output_path,
        "--model_path", best_model_path
    ]
    run_command(test_cmd)
    
    metrics = None
    hierarchical = None
    if os.path.exists(test_meta_json):
        with open(test_meta_json, "r") as f:
            meta = json.load(f)
        metrics = meta.get("metrics")
        hierarchical = meta.get("hierarchical")
        
    if metrics is None or hierarchical is None:
        raise ValueError(f"Failed to load metrics from {test_meta_json}")
        
    return recommender_run_name, metrics, hierarchical

# setup directories (main() points them to --outputs-dir and --models-dir)
def set_output_dirs(outputs_dir, models_dir):
    global OUTPUTS_DIR, MODELS_DIR, EXPERIMENT_DIR, CONFIGS_DIR, LOG_FILE, SID_REPORT_FILE, RETRIEVAL_REPORT_FILE
    OUTPUTS_DIR = outputs_dir
    MODELS_DIR = models_dir
    EXPERIMENT_DIR = os.path.join(OUTPUTS_DIR, "experiments")
    CONFIGS_DIR = os.path.join(EXPERIMENT_DIR, "configs")
    LOG_FILE = os.path.join(EXPERIMENT_DIR, "run.log")
    SID_REPORT_FILE = os.path.join(EXPERIMENT_DIR, "sid_evaluation_reports.md")
    RETRIEVAL_REPORT_FILE = os.path.join(EXPERIMENT_DIR, "seq2seq_retrieval_performance.md")
    os.makedirs(CONFIGS_DIR, exist_ok=True)

set_output_dirs("outputs", "models")

# copies an earlier run's semids.pt and Stage-1 meta JSONs into OUTPUTS_DIR, so Stage 2 runs on the same SIDs
def reuse_stage1_outputs(reuse_dir, exp_name, seed, semids_output_path):
    src_semids = os.path.join(reuse_dir, f"{exp_name.lower()}_seed_{seed}", "semids.pt")
    shutil.copy2(src_semids, semids_output_path)
    log_msg(f"Reusing Semantic IDs from {src_semids}")
    sid_report_text = ""
    for kind in ["quantizer", "generate"]:
        src_meta = os.path.join(reuse_dir, "experiments", f"{exp_name.lower()}_seed_{seed}_{kind}_meta.json")
        if os.path.exists(src_meta):
            shutil.copy2(src_meta, EXPERIMENT_DIR)
            with open(src_meta, "r") as f:
                sid_report_text = json.load(f).get("sid_report", "") or sid_report_text
    return sid_report_text

# helper to log messages both to run.log and stdout
def log_msg(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")

# run command and print/capture output in real-time
def run_command(cmd, env_vars=None):
    log_msg(f"Executing: {' '.join(cmd)}")
    
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
        
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        env=env
    )
    
    import io
    stdout_wrapper = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace", newline="")
    
    stdout_chars = []
    # read character by character to support multi-byte UTF-8 in real-time
    while True:
        char = stdout_wrapper.read(1)
        if not char:
            break
        sys.stdout.write(char)
        sys.stdout.flush()
        stdout_chars.append(char)
        
    process.wait()
    full_output = "".join(stdout_chars)
    
    if process.returncode != 0:
        log_msg(f"Command failed with return code {process.returncode}")
        raise subprocess.CalledProcessError(process.returncode, cmd, output=full_output)
        
    return full_output

# helper to initialize report files
def init_report_files():
    if not os.path.exists(SID_REPORT_FILE):
        with open(SID_REPORT_FILE, "w") as f:
            f.write("# Semantic ID Evaluation Reports\n")
            f.write("This file consolidates the SID Evaluation Reports for all configurations and seeds.\n\n")
            
    if not os.path.exists(RETRIEVAL_REPORT_FILE):
        with open(RETRIEVAL_REPORT_FILE, "w") as f:
            f.write("# Seq2Seq Recommender Retrieval Performance\n")
            f.write("This file compares the test set retrieval performance across different Semantic ID configurations.\n\n")

# Save a SID report to the consolidated Markdown file
def append_sid_report(label, report_text):
    with open(SID_REPORT_FILE, "a") as f:
        f.write(f"## {label}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("```text\n")
        f.write(report_text + "\n")
        f.write("```\n\n")
        f.write("---\n\n")

# Append individual and summary experiment metrics to report
def write_experiment_summary(exp_name, runs):
    # runs is a list of dicts: {"seed": seed, "metrics": metrics, "hierarchical": hierarchical, "run_name": name}
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_keys = ["Recall@5", "NDCG@5", "Recall@10", "NDCG@10", "Recall@20", "NDCG@20"]
    
    with open(RETRIEVAL_REPORT_FILE, "a") as f:
        f.write(f"## Experiment: {exp_name}\n\n")
        f.write("| Run Variant | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Run Name | Date |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        
        # Write individual seed rows
        for run in runs:
            m = run["metrics"]
            f.write(
                f"| {exp_name} (seed {run['seed']}) "
                f"| {m.get('Recall@5', 0.0):.4f} | {m.get('NDCG@5', 0.0):.4f} "
                f"| {m.get('Recall@10', 0.0):.4f} | {m.get('NDCG@10', 0.0):.4f} "
                f"| {m.get('Recall@20', 0.0):.4f} | {m.get('NDCG@20', 0.0):.4f} "
                f"| `{run['run_name']}` | {date_str} |\n"
            )
            
        # Compute stats if multiple seeds run
        if len(runs) > 1:
            stats = {}
            for key in metric_keys:
                vals = [r["metrics"].get(key, 0.0) for r in runs]
                mean = sum(vals) / len(vals)
                # Population SD (divides by n); the report uses the sample SD (n - 1) from report/scripts/aggregate.py.
                variance = sum((x - mean) ** 2 for x in vals) / len(vals)
                std_dev = math.sqrt(variance)
                stats[key] = (mean, std_dev)
                
            f.write(
                f"| **{exp_name} (Mean ± Std)** "
                f"| {stats['Recall@5'][0]:.4f} ± {stats['Recall@5'][1]:.4f} | {stats['NDCG@5'][0]:.4f} ± {stats['NDCG@5'][1]:.4f} "
                f"| {stats['Recall@10'][0]:.4f} ± {stats['Recall@10'][1]:.4f} | {stats['NDCG@10'][0]:.4f} ± {stats['NDCG@10'][1]:.4f} "
                f"| {stats['Recall@20'][0]:.4f} ± {stats['Recall@20'][1]:.4f} | {stats['NDCG@20'][0]:.4f} ± {stats['NDCG@20'][1]:.4f} "
                f"| Summary Statistics | {date_str} |\n"
            )
            
        f.write("\n### Detailed Hierarchical Metrics\n")
        # Write average hierarchical metrics if multiple, otherwise individual
        all_hier_keys = set()
        for r in runs:
            all_hier_keys.update(r["hierarchical"].keys())
        sorted_keys = sorted(list(all_hier_keys))
        
        for k in sorted_keys:
            vals = [r["hierarchical"].get(k, 0.0) for r in runs]
            mean = sum(vals) / len(vals)
            if len(runs) > 1:
                # Population SD, as above.
                variance = sum((x - mean) ** 2 for x in vals) / len(vals)
                std_dev = math.sqrt(variance)
                f.write(f"- {k}: {mean:.4f} ± {std_dev:.4f}\n")
            else:
                f.write(f"- {k}: {mean:.4f}\n")
                
        f.write("\n---\n\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Semantic ID experiments in an overlapping parallel pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick end-to-end check with minimal epochs")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated list of random seeds to run (e.g. 42,100,999)")
    parser.add_argument("--gpu-main", type=str, default=None, help="GPU index to use for the main tasks (recommender and active quantizer)")
    parser.add_argument("--gpu-quantizer", type=str, default=None, help="GPU index to use for background quantizer training")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel background quantizer training")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Folder for SIDs, experiment results, run configs and W&B runs (e.g. outputs_toys_v2)")
    parser.add_argument("--models-dir", type=str, default="models", help="Folder for quantizer and recommender checkpoints (e.g. models_toys_v2)")
    parser.add_argument("--reuse-semids", type=str, default=None, help="Skip Stage 1 and train Stage 2 on the SIDs of an earlier run's outputs folder (e.g. outputs_toys); its semids.pt and quantizer/generate meta JSONs are copied into --outputs-dir")
    args = parser.parse_args()
    
    reuse_dir = args.reuse_semids
    if reuse_dir is not None and os.path.abspath(reuse_dir) == os.path.abspath(args.outputs_dir):
        parser.error("--reuse-semids must differ from --outputs-dir, otherwise the earlier run's files would be overwritten")
    
    set_output_dirs(args.outputs_dir, args.models_dir)
    
    if args.gpu_main is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_main
        
    dry_run = args.dry_run
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    gpu_quant = args.gpu_quantizer
    parallel = not args.no_parallel and reuse_dir is None
    
    # Previous Amazon Beauty Experiments (commented out for reference)
    # beauty_experiments = [
    #     {
    #         "name": "RQ_KMeans",
    #         "quant_config": "config/rq-kmeans/amazon_beauty_rqkmeans.yaml",
    #         "recommender_config": "config/recommender/amazon_beauty_grid.yaml",
    #         "is_rqkmeans": True,
    #         "codebook_size": 256,
    #     },
    #     {
    #         "name": "FSQ",
    #         "quant_config": "config/fsq/amazon_beauty.yaml",
    #         "recommender_config": "config/recommender/amazon_beauty_grid_fsq.yaml",
    #         "codebook_size": 288,
    #     },
    #     {
    #         "name": "Residual_FSQ_Simple_HighDim",
    #         "quant_config": "config/fsq/amazon_beauty_rfsq_simple_highdim.yaml",
    #         "recommender_config": "config/recommender/amazon_beauty_grid_fsq.yaml",
    #         "codebook_size": 288,
    #     },
    #     {
    #         "name": "Residual_FSQ_Simple_LowDim",
    #         "quant_config": "config/fsq/amazon_beauty_rfsq_simple_lowdim.yaml",
    #         "recommender_config": "config/recommender/amazon_beauty_grid_fsq.yaml",
    #         "codebook_size": 288,
    #     },
    #     {
    #         "name": "Residual_FSQ_Complex",
    #         "quant_config": "config/fsq/amazon_beauty_rfsq_complex.yaml",
    #         "recommender_config": "config/recommender/amazon_beauty_grid_fsq.yaml",
    #         "codebook_size": 288,
    #     },
    #     {
    #         "name": "RQ_VAE",
    #         "quant_config": "config/rq-vae/amazon_beauty_tiger.yaml",
    #         "recommender_config": "config/recommender/amazon_beauty_grid.yaml",
    #         "codebook_size": 256,
    #     }
    # ]

    # Amazon Toys Experiments
    # Stage 2 with TIGER's 100k-step budget and no early stopping (*_100k.yaml); the reported Toys runs used
    # amazon_toy_grid.yaml / amazon_toy_grid_fsq.yaml (early stopping, patience 15)
    experiments = [
        {
            "name": "RQ_KMeans",
            "quant_config": "config/rq-kmeans/amazon_toy_rqkmeans.yaml",
            "recommender_config": "config/recommender/amazon_toy_grid_100k.yaml",
            "is_rqkmeans": True,
            "codebook_size": 256,
        },
        {
            "name": "FSQ",
            "quant_config": "config/fsq/amazon_toy.yaml",
            "recommender_config": "config/recommender/amazon_toy_grid_fsq_100k.yaml",
            "codebook_size": 288,
        },
        {
            "name": "Residual_FSQ_Simple_HighDim",
            "quant_config": "config/fsq/amazon_toy_rfsq_simple_highdim.yaml",
            "recommender_config": "config/recommender/amazon_toy_grid_fsq_100k.yaml",
            "codebook_size": 288,
        },
        {
            "name": "Residual_FSQ_Simple_LowDim",
            "quant_config": "config/fsq/amazon_toy_rfsq_simple_lowdim.yaml",
            "recommender_config": "config/recommender/amazon_toy_grid_fsq_100k.yaml",
            "codebook_size": 288,
        },
        {
            "name": "Residual_FSQ_Complex",
            "quant_config": "config/fsq/amazon_toy_rfsq_complex.yaml",
            "recommender_config": "config/recommender/amazon_toy_grid_fsq_100k.yaml",
            "codebook_size": 288,
        },
        {
            "name": "RQ_VAE",
            "quant_config": "config/rq-vae/amazon_toy_tiger.yaml",
            "recommender_config": "config/recommender/amazon_toy_grid_100k.yaml",
            "codebook_size": 256,
        }
    ]
    
    if reuse_dir is not None:
        missing = [os.path.join(reuse_dir, f"{exp['name'].lower()}_seed_{seed}", "semids.pt") for exp in experiments for seed in seeds]
        missing = [p for p in missing if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"--reuse-semids: missing Semantic IDs: {missing}")
    
    init_report_files()
    
    log_msg(f"Starting experiments run. Seeds = {seeds}, Dry run = {dry_run}, Parallel = {parallel}, Reuse SIDs from = {reuse_dir}")
    
    # Flatten experiments and seeds into a sequence of pipeline tasks
    tasks = []
    for exp in experiments:
        for seed in seeds:
            tasks.append((exp, seed))
            
    results_by_exp = {exp["name"]: [] for exp in experiments}
    
    # Track the active background quantizer job details
    bg_proc = None
    bg_task_idx = None
    bg_log_file = None
    bg_log_path = None
    bg_cmd = None
    
    # Process tasks list
    for j in range(len(tasks)):
        exp, seed = tasks[j]
        exp_name = exp["name"]
        
        log_msg(f"\n========================================\n[START] Task {j+1}/{len(tasks)}: {exp_name} (Seed {seed})\n========================================")
        
        # Paths for temporary JSON metadata configurations
        quant_meta_json = os.path.join(EXPERIMENT_DIR, f"{exp_name.lower()}_seed_{seed}_quantizer_meta.json")
        gen_meta_json = os.path.join(EXPERIMENT_DIR, f"{exp_name.lower()}_seed_{seed}_generate_meta.json")
        train_meta_json = os.path.join(EXPERIMENT_DIR, f"{exp_name.lower()}_seed_{seed}_train_meta.json")
        test_meta_json = os.path.join(EXPERIMENT_DIR, f"{exp_name.lower()}_seed_{seed}_test_meta.json")
        
        # Clean up any leftover temp files from prior runs (skip quant_meta_json if it was populated in the background)
        clean_paths = [gen_meta_json, train_meta_json, test_meta_json]
        if not (bg_proc is not None and bg_task_idx == j):
            clean_paths.append(quant_meta_json)
            
        for path in clean_paths:
            if os.path.exists(path):
                os.remove(path)

        try:
            # Step 1: Secure or wait for the trained quantizer / semantic IDs
            quantizer_model_path = None
            semids_output_path = os.path.join(OUTPUTS_DIR, f"{exp_name.lower()}_seed_{seed}", "semids.pt")
            os.makedirs(os.path.dirname(semids_output_path), exist_ok=True)
            
            sid_report_text = ""
            
            # Stage 1 skipped: reuse the SIDs of an earlier run
            if reuse_dir is not None:
                sid_report_text = reuse_stage1_outputs(reuse_dir, exp_name, seed, semids_output_path)
                if sid_report_text:
                    append_sid_report(f"{exp_name} (seed {seed}, reused from {reuse_dir})", sid_report_text)
            
            # Check if this task's quantizer was already started in the background
            elif bg_proc is not None and bg_task_idx == j:
                log_msg(f"Waiting for background quantizer training of Task {j+1} ({exp_name}, seed {seed}) to complete...")
                bg_proc.wait()
                if bg_log_file:
                    bg_log_file.close()
                    
                if bg_proc.returncode != 0:
                    log_msg(f"Background quantizer job for Task {j+1} failed with code {bg_proc.returncode}")
                    raise subprocess.CalledProcessError(bg_proc.returncode, bg_cmd)
                    
                # Read background quantizer results from JSON metadata
                if os.path.exists(quant_meta_json):
                    with open(quant_meta_json, "r") as f:
                        meta = json.load(f)
                    quantizer_model_path = meta.get("model_path")
                    sid_report_text = meta.get("sid_report", "")
                    if sid_report_text:
                        append_sid_report(f"{exp_name} (seed {seed})", sid_report_text)
                        log_msg(f"Saved background RQ-KMeans SID report for seed {seed}")
                else:
                    log_msg(f"WARNING: Background quantizer metadata JSON not found at: {quant_meta_json}")
                    
                bg_proc = None
                bg_task_idx = None
                
            else:
                # If we couldn't run it in the background, train it synchronously now
                cmd, _, _, _ = prepare_quantizer_task(exp, seed, dry_run)
                run_command(cmd)
                if os.path.exists(quant_meta_json):
                    with open(quant_meta_json, "r") as f:
                        meta = json.load(f)
                    quantizer_model_path = meta.get("model_path")
                    sid_report_text = meta.get("sid_report", "")
                    if exp.get("is_rqkmeans") and sid_report_text:
                        append_sid_report(f"{exp_name} (seed {seed})", sid_report_text)
            
            # Step 2: Generate Semantic IDs (if not RQ-KMeans, which creates them directly, or reused)
            if not exp.get("is_rqkmeans") and reuse_dir is None:
                if not quantizer_model_path:
                    quantizer_model_path = os.path.join(MODELS_DIR, "quantizer", f"{exp_name.lower()}_seed_{seed}.pt")
                    
                if not os.path.exists(quantizer_model_path):
                    raise FileNotFoundError(f"Expected quantizer model checkpoint not found at: {quantizer_model_path}")
                    
                gen_overrides = {
                    "general.seed": seed,
                    "general.meta_json": gen_meta_json
                }
                gen_run_config = os.path.join(CONFIGS_DIR, f"{exp_name}_seed_{seed}_generate.yaml")
                prepare_run_config(exp["quant_config"], gen_run_config, gen_overrides)
                
                cmd = ["python", "main.py", "generate-ids", "--config", gen_run_config, "--model_path", quantizer_model_path, "--output_path", semids_output_path]
                run_command(cmd)
                if os.path.exists(gen_meta_json):
                    with open(gen_meta_json, "r") as f:
                        meta = json.load(f)
                    sid_report_text = meta.get("sid_report", "")
                    if sid_report_text:
                        append_sid_report(f"{exp_name} (seed {seed})", sid_report_text)
                        log_msg(f"Saved SID Evaluation Report for {exp_name} (seed {seed})")
                
            # Step 3: Launch the NEXT task's quantizer in the background before starting Seq2Seq training
            if parallel and (j + 1) < len(tasks):
                next_exp, next_seed = tasks[j + 1]
                next_exp_name = next_exp["name"]
                next_quant_meta_json = os.path.join(EXPERIMENT_DIR, f"{next_exp_name.lower()}_seed_{next_seed}_quantizer_meta.json")
                if os.path.exists(next_quant_meta_json):
                    os.remove(next_quant_meta_json)
                
                log_msg(f"Launching background quantizer training for Task {j+2} ({next_exp_name}, seed {next_seed})...")
                bg_cmd, _, _, _ = prepare_quantizer_task(next_exp, next_seed, dry_run)
                
                bg_env = os.environ.copy()
                if gpu_quant is not None:
                    bg_env["CUDA_VISIBLE_DEVICES"] = gpu_quant
                    log_msg(f"Targeting background quantizer to GPU index {gpu_quant}")
                    
                bg_log_path = os.path.join(EXPERIMENT_DIR, f"{next_exp_name}_seed_{next_seed}_quantizer.log")
                bg_log_file = open(bg_log_path, "w")
                bg_proc = subprocess.Popen(bg_cmd, stdout=bg_log_file, stderr=subprocess.STDOUT, env=bg_env)
                bg_task_idx = j + 1
                    
            # Step 4 & 5: Train & Test Seq2Seq Recommender
            recommender_run_name, metrics, hierarchical = run_recommender_task(exp, seed, semids_output_path, dry_run)
            
            results_by_exp[exp_name].append({
                "seed": seed,
                "metrics": metrics,
                "hierarchical": hierarchical,
                "run_name": recommender_run_name
            })
            log_msg(f"Successfully completed Task {j+1}. Metrics: {metrics}")
            
            # If this is the last seed for this experiment, compute statistics and write to report
            if len(results_by_exp[exp_name]) == len(seeds):
                write_experiment_summary(exp_name, results_by_exp[exp_name])
                log_msg(f"Successfully compiled consolidated retrieval reports for: {exp_name}")
                
        except Exception as e:
            log_msg(f"ERROR: Task {j+1} ({exp_name}, seed {seed}) failed. Reason: {e}")
            if bg_proc is not None and bg_task_idx == j:
                try:
                    bg_proc.kill()
                except:
                    pass
                bg_proc = None
                bg_task_idx = None
            continue
            
    log_msg("All experiments finished.")

if __name__ == "__main__":
    main()
