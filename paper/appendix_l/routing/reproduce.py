"""Recompute Appendix L, Table 19, from per-receiver router measurements."""
import argparse
import csv
import gzip
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
WINDOWS = ("last_token", "suffix_k", "divergence_k", "full_step")
ARMS = ("router_logits", "all_router")
MODELS = ("Qwen3.6", "Qwen3", "GLM-4.7-Flash", "Lightning", "Nano", "Gemma4", "OLMoE", "GPT-OSS-20B", "GPT-OSS-120B")
COLUMNS = ("model", "topk_changed", "matched_paired_kl_min", "matched_paired_kl_max", "single_layer_donor_e_c_kl", "all_router_donor_e_c_kl")
D_RUNS = (("local_d", "last_token", "router_logits"), ("wide_d", "full_step", "all_router"))


class ConditionMean(NamedTuple):
    receiver_count: int
    changed: float
    kl: float


def load_cells():
    cells = defaultdict(dict)
    with gzip.open(HERE / "observations.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[name] for name in ("model", "run", "window", "arm", "condition"))
            receiver = (row["pair_id"], row["receiver_role"])
            changed = float(row["topk_changed"])
            kl = float(row["router_kl"])
            if receiver in cells[key] or not 0 <= changed <= 1 or not math.isfinite(kl):
                raise ValueError(f"Invalid receiver record in {key}: {receiver}")
            cells[key][receiver] = (changed, kl)

    conditions = list(itertools.product(("windows",), WINDOWS, ARMS, ("paired", "matched")))
    conditions.extend((*setting, donor) for setting in D_RUNS for donor in ("donor_C", "donor_E"))
    if set(cells) != {(model, *cell) for model in MODELS for cell in conditions}:
        raise ValueError("Missing or unexpected model, window, intervention or donor condition")

    means = {}
    for key, traces in cells.items():
        if (key[1] != "windows" or key[2] == "last_token") and len(traces) != 580:
            raise ValueError(f"Expected 580 receiver traces in {key}, found {len(traces)}")
        means[key] = ConditionMean(len(traces), fmean(value[0] for value in traces.values()),
                                   fmean(value[1] for value in traces.values()))
    return means


def table_rows(means):
    for model in MODELS:
        ratios = [means[model, "windows", window, arm, "matched"].kl /
                  means[model, "windows", window, arm, "paired"].kl
                  for window, arm in itertools.product(WINDOWS, ARMS)]
        changed = means[model, "windows", "last_token", "router_logits", "paired"].changed
        donor_ratios = []
        for run, window, arm in D_RUNS:
            donor_ratios.append(means[model, run, window, arm, "donor_E"].kl /
                                means[model, run, window, arm, "donor_C"].kl)
        yield [model, *(f"{value:.3f}" for value in (changed, min(ratios), max(ratios), *donor_ratios))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Directory for table19.csv and condition_means.csv")
    args = parser.parse_args()
    means = load_cells()
    table = [list(COLUMNS), *table_rows(means)]
    with (HERE / "table19_reference.csv").open(newline="") as stream:
        if table != list(csv.reader(stream)):
            raise ValueError("A displayed number differs from the manuscript")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "table19.csv").open("w", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(table)
        with (args.output / "condition_means.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("model", "run", "window", "arm", "condition", "receiver_count",
                             "mean_topk_changed", "mean_router_kl"))
            writer.writerows((*key, *means[key]) for key in sorted(means))
    else:
        csv.writer(sys.stdout, lineterminator="\n").writerows(table)


if __name__ == "__main__":
    main()
