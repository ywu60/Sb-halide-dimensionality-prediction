"""GPT baselines on the analysis-3 Sb-halide dataset, same held-out test split as the ML models.

Adapted from analysis 2's 03_run_llm.py:
  - target is target_non0D (0 = 0D, 1 = non-0D) -- FLIPPED vs. analysis 2's target_0D (1 = 0D). The
    prompt instruction, shot labels, and probability columns are all updated to match.
  - the model is only ever asked to emit a single visible character, "0" or "1"; token-level
    log-probabilities for exactly those two tokens are converted into probability_0D / probability_non0D
    wherever the API returns logprobs (gpt-4.1). Reasoning models return a label only.

    export OPENAI_API_KEY=...
    python 05_run_llm.py --data prepared_data.xlsx --out results_llm
"""
import argparse, json, math, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score

HALIDES = ["F", "Cl", "Br", "I"]

# Standard API text-token prices per million tokens. Keep this table deliberately explicit:
# a blank entry means the script preserves usage but does not present a misleading cost estimate.
# Cache-write fees are excluded because Chat Completions usage does not report cache-write tokens.
MODEL_TOKEN_PRICES_USD_PER_MILLION = {
    "gpt-6-astra": {"input": 10.00, "cached_input": 1.00, "output": 50.00},
}

def usage_and_cost(usage: dict, model: str):
    details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cached_input_tokens = int(details.get("cached_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
    price = MODEL_TOKEN_PRICES_USD_PER_MILLION.get(model)
    estimated_cost = np.nan
    if price:
        estimated_cost = ((prompt_tokens - cached_input_tokens) * price["input"] + cached_input_tokens * price["cached_input"] + completion_tokens * price["output"]) / 1_000_000
    return {
        "prompt_tokens": prompt_tokens,
        "cached_input_tokens": cached_input_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "estimated_standard_cost_usd": estimated_cost,
    }

def api_object_to_dict(value):
    """Support both current OpenAI SDK/Pydantic and the older environment used for this analysis."""
    try:
        return value.model_dump()
    except (AttributeError, TypeError):
        # openai==1.55 with this environment's Pydantic raises from both model_dump()
        # and dict(). Its public model fields are nevertheless safely available in __dict__.
        def plain(x):
            if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)): return [plain(v) for v in x]
            if hasattr(x, "__dict__"): return {k: plain(v) for k, v in vars(x).items() if not k.startswith("_")}
            return x
        return plain(value)

def select_12_shots(train: pd.DataFrame):
    # RDKit is only required for diversity-aware 12-shot selection. Keeping this import local
    # lets zero-shot and all-shot API evaluations run in a lightweight inference environment.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024); fps = {}
    for i, s in zip(train.index, train.canonical_cation_smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None: raise ValueError(f"Invalid SMILES for shot selection: {s}")
        fps[i] = gen.GetFingerprint(mol)
    chosen_by_label = {}
    for label in [0, 1]:
        candidates = list(train.index[train.target_non0D.eq(label)]); chosen = []
        if len(candidates) < 6: raise ValueError(f"Need at least 6 training examples for target_non0D={label}")
        while len(chosen) < 6:
            prior = train.loc[chosen] if chosen else train.iloc[0:0]; best = None
            for i in candidates:
                if i in chosen: continue
                r = train.loc[i]; sim = max((DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in chosen), default=0.0)
                score = 1.0 - sim
                score += .45 * int(str(r.sb_oxidation_state) not in set(prior.sb_oxidation_state.astype(str)))
                score += .35 * int(str(r.Sb_halide) not in set(prior.Sb_halide.astype(str)))
                score += .20 * int(("/" in str(r.Sb_halide)) not in set(prior.Sb_halide.astype(str).str.contains("/")))
                candidate = (score, str(r.compound_id), i)
                if best is None or candidate[:2] > best[:2]: best = candidate
            chosen.append(best[2])
        chosen_by_label[label] = chosen
    # Interleave 0D / non-0D so shots are NOT grouped by class. Presenting all of one class as the final block
    # induces strong recency bias (the model tends to echo the last examples' label); alternating removes it.
    interleaved = [i for pair in zip(chosen_by_label[0], chosen_by_label[1]) for i in pair]
    shots = train.loc[interleaved].copy(); shots["shot_class"] = shots.target_non0D.map({0: "0D", 1: "non-0D"})
    return shots

def order_all_shots(train: pd.DataFrame, seed: int, limit: int | None):
    # The whole train split as examples. Classes are spread evenly across the sequence rather than plainly
    # shuffled: with the class imbalance a shuffle leaves long single-class runs, and a single-class tail is
    # exactly the recency bias select_12_shots interleaves to avoid. Each class is laid on the same [0, 1]
    # interval and the two are merged, so the minority class stays uniformly distributed at any imbalance.
    rng = np.random.default_rng(seed); positions = []
    for label in [0, 1]:
        idx = rng.permutation(train.index[train.target_non0D.eq(label)].to_numpy())
        if limit is not None: idx = idx[:max(1, round(limit * len(idx) / len(train)))]
        positions += [((rank + .5) / len(idx), label, i) for rank, i in enumerate(idx)]
    order = [i for _, _, i in sorted(positions, key=lambda t: (t[0], t[1]))]
    shots = train.loc[order].copy(); shots["shot_class"] = shots.target_non0D.map({0: "0D", 1: "non-0D"})
    return shots

def chemical_block(r: pd.Series, representation: str):
    hs = ", ".join(f"{h}={r[f'inorganic_{h}_fraction']:.4f}" for h in HALIDES)
    ratio = r.get("inorganic_halide_per_metal", float("nan"))
    ratio_txt = f"{ratio:.2f}" if pd.notna(ratio) else "unknown (variable formula)"
    value = r.organic_component if representation == "name" else r.canonical_cation_smiles
    label = "Organic-cation IUPAC name" if representation == "name" else "Organic-cation SMILES"
    water_txt = f"{r.water_count:g}" if pd.notna(r.water_count) else "unknown"
    return f"Normalized formula: {r.formula_normalized}\nSb oxidation state: {r.sb_oxidation_state}\nWater molecules per formula unit: {water_txt}\nInorganic Sb-halide fractions: {hs}\nInorganic halide-to-Sb(+Bi) ratio: {ratio_txt}\n{label}: {value}"

def prompt_for(r: pd.Series, representation: str, shots: pd.DataFrame | None):
    instruction = """Classify only the dimensionality of inorganic Sb-halide connectivity. Connections
    count only when Sb centers are joined through inorganic halide atoms. Ignore whole-crystal
    packing, organic networks, hydrogen bonds, and supramolecular dimensionality. A discrete
    Sb-halide anion or finite cluster is 0D; an infinite chain, layer, or framework is non-0D.
    Labels: 0 = 0D; 1 = non-0D. Reason internally from formula, Sb oxidation state, water content,
    inorganic halide composition, the inorganic halide-to-metal ratio, and the organic cation.
    Your entire visible answer must be exactly one character: 0 or 1."""
    examples = ""
    # The example block sits before the only part that varies across test compounds, so the long all-shot
    # prefix is identical call to call and OpenAI's automatic prompt caching covers it.
    if shots is not None: examples = "\n\nLabeled examples:\n" + "\n\n".join(chemical_block(x, representation) + f"\nAnswer: {int(x.target_non0D)}" for _, x in shots.iterrows())
    return instruction + examples + "\n\nCompound to classify:\n" + chemical_block(r, representation) + "\nAnswer:"

def prompt_messages(r: pd.Series, representation: str, shots: pd.DataFrame | None):
    prompt = prompt_for(r, representation, shots)
    if shots is None:
        return [{"role": "user", "content": prompt}]
    # Astra's cache matches explicit content-block boundaries. The first block is identical
    # across all test rows; only the second block varies by compound.
    marker = "\n\nCompound to classify:\n"
    prefix, suffix = prompt.rsplit(marker, 1)
    return [{"role": "user", "content": [
        {"type": "text", "text": prefix, "prompt_cache_breakpoint": {"mode": "explicit"}},
        {"type": "text", "text": marker + suffix},
    ]}]

def supports_logprobs(model: str):
    return model.startswith("gpt-4.1")

class QuotaExhausted(RuntimeError):
    """Billing quota is gone. Not retryable: every further call fails identically, so abort the whole run."""

def is_quota_error(e: Exception):
    return getattr(e, "code", None) == "insufficient_quota" or "insufficient_quota" in str(e)

def prompt_encoding():
    return tiktoken.get_encoding("o200k_base")  # only used to report prompt size, so an exact model mapping is unnecessary

def class_token_bias(model: str):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Older tiktoken releases may not map dated GPT-4.1 snapshot names.
        encoding = tiktoken.get_encoding("o200k_base")
    token_ids = set()
    for text in ["0", "1", " 0", " 1"]:
        ids = encoding.encode(text)
        if len(ids) == 1: token_ids.add(ids[0])
    if len(token_ids) < 2: raise RuntimeError(f"Could not resolve single-token 0/1 forms for {model}")
    return {str(token_id): 100 for token_id in token_ids}

def one_token_result(client: OpenAI, model: str, messages, reasoning_effort: str, reasoning_max_tokens: int, prompt_cache_key: str | None = None, retries=5):
    last = None; budget = reasoning_max_tokens  # for reasoning models this budget covers hidden reasoning + the visible digit
    for attempt in range(retries):
        try:
            params = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": 1024,
            }
            if supports_logprobs(model):
                params |= {"logprobs": True, "top_logprobs": 20, "temperature": 1, "max_completion_tokens": 1, "logit_bias": class_token_bias(model)}
            else:
                # Older OpenAI Python SDKs do not yet expose reasoning_effort as a typed
                # Chat Completions argument. extra_body preserves the identical wire request.
                extra_body = {"reasoning_effort": reasoning_effort}
                if prompt_cache_key:
                    extra_body["prompt_cache_key"] = prompt_cache_key
                    extra_body["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
                params |= {"extra_body": extra_body, "max_completion_tokens": budget}
            response = client.chat.completions.create(**params)
            choice = response.choices[0]; answer = (choice.message.content or "").strip()
            if answer not in {"0", "1"}:
                usage = api_object_to_dict(response.usage) if response.usage else {}
                details = usage.get("completion_tokens_details") or {}
                # A reasoning model can spend the whole budget on hidden reasoning and return no visible digit
                # (finish_reason == "length"). Grow the budget so the retry has room to emit the answer.
                if not supports_logprobs(model) and choice.finish_reason == "length": budget = min(budget * 2, 32768)
                raise RuntimeError(
                    f"Expected exactly 0 or 1, received {choice.message.content!r}; "
                    f"finish_reason={choice.finish_reason!r}, completion_tokens={usage.get('completion_tokens')}, "
                    f"reasoning_tokens={details.get('reasoning_tokens')}, max_completion_tokens={params['max_completion_tokens']}"
                )
            result = {
                "prediction": int(answer),
                "visible_answer": int(answer),
                "model_resolved": response.model,
                "usage": api_object_to_dict(response.usage),
                "has_token_probabilities": False,
                "probability_method": "not_available",
                "raw_probability_token_0": np.nan,
                "raw_probability_token_1": np.nan,
                "probability_0D_conditional_01": np.nan,
                "probability_non0D_conditional_01": np.nan,
                "captured_probability_mass_01": np.nan,
                "logprob_0": np.nan,
                "logprob_1": np.nan,
            }
            if not supports_logprobs(model): return result
            items = choice.logprobs.content or []; item = next((x for x in items if x.token.strip() == answer), None)
            if item is None: raise RuntimeError("No logprobs for the answer token")
            alternatives = {x.token: x.logprob for x in item.top_logprobs}
            alternatives[item.token] = item.logprob
            class_logprobs = {label: [lp for token, lp in alternatives.items() if token.strip() == label] for label in ["0", "1"]}
            missing = [label for label, values in class_logprobs.items() if not values]
            if len(missing) == 2: raise RuntimeError(f"Neither class token was returned: {class_logprobs}")
            if missing:
                # The model was confident enough that the losing class fell below the top-20 logprob cutoff even
                # with the equal bias. Its true mass is below the least-likely returned token, so floor it there
                # rather than discarding the prediction (yields a near-1/near-0 probability, which is correct).
                floor = min((x.logprob for x in item.top_logprobs), default=item.logprob)
                for label in missing: class_logprobs[label] = [floor]
            raw0 = sum(math.exp(lp) for lp in class_logprobs["0"]); raw1 = sum(math.exp(lp) for lp in class_logprobs["1"]); denom = raw0 + raw1
            lp0 = math.log(raw0); lp1 = math.log(raw1)
            # token "0" = 0D, token "1" = non-0D under the analysis-3 label convention.
            result |= {
                "prediction": int(raw1 >= raw0),
                "has_token_probabilities": True,
                "probability_method": "equal_logit_bias_conditional_01",
                "raw_probability_token_0": raw0,
                "raw_probability_token_1": raw1,
                "probability_0D_conditional_01": raw0 / denom,
                "probability_non0D_conditional_01": raw1 / denom,
                "captured_probability_mass_01": denom,
                "logprob_0": lp0,
                "logprob_1": lp1,
            }
            return result
        except Exception as e:
            # A quota 429 is not a rate limit -- retrying it just burns the retry budget on every remaining call.
            if is_quota_error(e): raise QuotaExhausted(str(e)) from e
            last = e
            print(f"    Attempt {attempt + 1}/{retries} failed: {type(e).__name__}: {e}", flush=True)
            # Capped exponential backoff: concurrent runs hit 429s, which need longer waits than a bad-answer retry.
            if attempt + 1 < retries: time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Failed for {model}: {last}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results_llm")
    ap.add_argument("--models", nargs="+", default=["gpt-4.1-2025-04-14", "gpt-5.1-2025-11-13", "gpt-5.5-2026-04-23", "gpt-5.6-sol", "gpt-6-astra"]); ap.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--reasoning-max-tokens", type=int, default=16384, help="Completion-token budget for reasoning models; includes hidden reasoning tokens")
    ap.add_argument("--timeout", type=float, default=180, help="Per-request timeout in seconds")
    ap.add_argument("--prompting", nargs="+", default=["zero", "12shot", "allshot"], choices=["zero", "12shot", "allshot"], help="Which prompting conditions to run; allshot puts the entire train split in the prompt")
    ap.add_argument("--all-shot-max", type=int, help="Cap the number of all-shot examples (class ratio preserved); default is the full train split")
    ap.add_argument("--selected-12-shots", help="Reuse a previously saved selected_12_shots.xlsx workbook, preserving an identical few-shot baseline without requiring RDKit")
    ap.add_argument("--seed", type=int, default=42, help="Seed for the all-shot example ordering")
    ap.add_argument("--concurrency", type=int, default=8, help="Parallel API calls within one model/representation/prompting condition")
    ap.add_argument("--limit", type=int); ap.add_argument("--test-offset", type=int, default=0, help="Skip this many test rows before applying --limit; useful for a cache-verification slice")
    ap.add_argument("--representations", nargs="+", choices=["name", "smiles"], default=["name", "smiles"], help="Input representations to evaluate")
    ap.add_argument("--compound-ids", nargs="+", help="Restrict evaluation to specific test compound IDs")
    ap.add_argument("--fresh", action="store_true"); ap.add_argument("--fail-fast", action="store_true"); a = ap.parse_args()
    if "OPENAI_API_KEY" not in os.environ: raise RuntimeError("Set OPENAI_API_KEY before running this script")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); client = OpenAI(timeout=a.timeout, max_retries=0)
    print(f"Loading {a.data} ...", flush=True); df = pd.read_excel(a.data); print(f"Loaded {len(df)} rows.", flush=True)
    required = {"compound_id", "split", "target_non0D", "formula_normalized", "sb_oxidation_state", "water_count", "organic_component", "canonical_cation_smiles", "inorganic_halide_per_metal"} | {f"inorganic_{h}_fraction" for h in HALIDES}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Prepared-data columns missing: {sorted(missing)}")
    train = df[df.split.eq("train")].copy(); test = df[df.split.eq("test")].copy()
    if a.compound_ids: test = test[test.compound_id.astype(str).isin(a.compound_ids)].copy()
    shot_columns = [c for c in df.columns if not c.startswith("smi_ted_")]
    shots = None
    if "12shot" in a.prompting:
        if a.selected_12_shots:
            print(f"Loading saved 12-shot examples from {a.selected_12_shots} ...", flush=True)
            shots = pd.read_excel(a.selected_12_shots)
            required_shot_columns = {"compound_id", "target_non0D", "shot_class"} | required
            missing_shot_columns = required_shot_columns - set(shots.columns)
            if missing_shot_columns: raise ValueError(f"Saved 12-shot workbook is missing columns: {sorted(missing_shot_columns)}")
            if len(shots) != 12 or set(shots.target_non0D) != {0, 1}: raise ValueError("Saved 12-shot workbook must contain 12 examples from both classes")
        else:
            print("Selecting balanced 12-shot examples ...", flush=True); shots = select_12_shots(train)
        shots[shot_columns + ["shot_class"]].to_excel(out / "selected_12_shots.xlsx", index=False)
    all_shots = None
    if "allshot" in a.prompting:
        all_shots = order_all_shots(train, a.seed, a.all_shot_max)
        all_shots[shot_columns + ["shot_class"]].to_excel(out / "all_shot_examples.xlsx", index=False)
        n_0d = int(all_shots.target_non0D.eq(0).sum()); size = len(prompt_encoding().encode(prompt_for(test.iloc[0], "smiles", all_shots)))
        print(f"All-shot: {len(all_shots)} train examples ({n_0d} 0D / {len(all_shots) - n_0d} non-0D), ~{size} prompt tokens per call", flush=True)
    test = test.iloc[a.test_offset:]
    if a.limit: test = test.head(a.limit)
    conditions = [(name, {"zero": None, "12shot": shots, "allshot": all_shots}[name]) for name in a.prompting]
    checkpoint = out / "llm_predictions_checkpoint.csv"; rows = [] if a.fresh or not checkpoint.exists() else pd.read_csv(checkpoint).to_dict("records")
    done = {(str(x["compound_id"]), str(x["model_requested"]), str(x["representation"]), str(x["prompting"])) for x in rows if x.get("status") == "ok"}
    failures = []; total = len(a.models) * len(a.representations) * len(conditions) * len(test); call_number = 0
    # Count only the keys this run plans to make; len(done) also covers checkpointed conditions outside the
    # current --models / --prompting selection, which used to make the progress denominator negative.
    planned = [(str(r.compound_id), model, representation, shot_name) for model in a.models for representation in a.representations for shot_name, _ in conditions for _, r in test.iterrows()]
    remaining = sum(key not in done for key in planned)
    print(f"Planned evaluations: {total}; already completed in checkpoint: {total - remaining}; to run: {remaining}", flush=True)
    lock = threading.Lock(); aborted = threading.Event()
    def evaluate(job):
        # Runs on a worker thread. The API call happens outside the lock; only the shared bookkeeping and the
        # checkpoint write are serialized. One log line per completed call, since interleaved start/finish pairs
        # are unreadable once calls overlap.
        nonlocal call_number
        if aborted.is_set(): return  # quota died on another thread; nothing further can succeed
        model, representation, shot_name, examples, r = job
        key = (str(r.compound_id), model, representation, shot_name)
        effort = "none" if supports_logprobs(model) else a.reasoning_effort
        try:
            cache_key = f"sbhalide-{model}-{representation}-{shot_name}-s{a.seed}"
            result = one_token_result(client, model, prompt_messages(r, representation, examples), a.reasoning_effort, a.reasoning_max_tokens, cache_key)
            row = {"compound_id": r.compound_id, "model_requested": model, "model_resolved": result["model_resolved"], "representation": representation, "prompting": shot_name, "reasoning_effort": effort, "true": int(r.target_non0D), "prediction": result["prediction"], "visible_answer": result["visible_answer"], "has_token_probabilities": result["has_token_probabilities"], "probability_method": result["probability_method"], "raw_probability_token_0": result["raw_probability_token_0"], "raw_probability_token_1": result["raw_probability_token_1"], "probability_0D_conditional_01": result["probability_0D_conditional_01"], "probability_non0D_conditional_01": result["probability_non0D_conditional_01"], "captured_probability_mass_01": result["captured_probability_mass_01"], "logprob_0": result["logprob_0"], "logprob_1": result["logprob_1"], **usage_and_cost(result["usage"], model), "usage": json.dumps(result["usage"]), "status": "ok", "error": ""}
            prob_text = f", P(0D|0/1)={result['probability_0D_conditional_01']:.4f}" if result["has_token_probabilities"] else ""
            outcome = f"prediction={result['prediction']}{prob_text}"; error = None
        except QuotaExhausted as e:
            # Don't checkpoint a row for this: it carries no result, and a clean checkpoint means the resume
            # needs no manual cleanup.
            if not aborted.is_set():
                aborted.set()
                print(f"\n*** ABORTING: OpenAI quota exhausted. {e}\n*** Banked results are intact; top up and rerun the same command to resume.\n", flush=True)
            return
        except Exception as e:
            row = {"compound_id": r.compound_id, "model_requested": model, "model_resolved": "", "representation": representation, "prompting": shot_name, "reasoning_effort": effort, "true": int(r.target_non0D), "prediction": np.nan, "status": "error", "error": str(e)}
            outcome = f"ERROR: {type(e).__name__}: {e}"; error = e
        with lock:
            call_number += 1; rows.append(row)
            (done.add if error is None else failures.append)(key)
            print(f"[{call_number}/{remaining}] {model} | {r.compound_id} | {representation} | {shot_name} | {outcome}", flush=True)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
        if error is not None and a.fail_fast: raise error

    for model in a.models:
        for representation in a.representations:
            for shot_name, examples in conditions:
                if aborted.is_set(): break
                jobs = [(model, representation, shot_name, examples, r) for _, r in test.iterrows() if (str(r.compound_id), model, representation, shot_name) not in done]
                if not jobs: continue
                mode = "token probabilities" if supports_logprobs(model) else f"label only, reasoning={a.reasoning_effort}"
                print(f"--- {model} | {representation} | {shot_name} | {mode} | {len(jobs)} calls, concurrency={a.concurrency} ---", flush=True)
                # Prime the shared prompt prefix with one call before fanning out: launching N workers at once
                # would have all N miss the cache, which costs real money on the ~32k-token all-shot prompt.
                evaluate(jobs[0])
                if len(jobs) > 1 and not aborted.is_set():
                    with ThreadPoolExecutor(max_workers=a.concurrency) as pool: list(pool.map(evaluate, jobs[1:]))
            if aborted.is_set(): break
        if aborted.is_set(): break
    pred = pd.DataFrame(rows)
    if pred.empty:
        if aborted.is_set():
            raise SystemExit("Run aborted on quota exhaustion before any evaluation completed; top up and rerun the same command to resume.")
        raise RuntimeError("No evaluations were completed")
    good = pred[pred.status.eq("ok")].copy(); summary_rows = []
    # The checkpoint dedups on model_requested but the summary groups on model_resolved, so two aliases for one
    # model (e.g. "gpt-5.6" and "gpt-5.6-sol") would double-count the compounds they share. Drop those here.
    before = len(good); good = good.drop_duplicates(["compound_id", "model_resolved", "representation", "prompting"], keep="last")
    if len(good) < before: print(f"Dropped {before - len(good)} duplicate compound/model/condition rows before scoring", flush=True)
    for keys, x in good.groupby(["model_resolved", "representation", "prompting", "reasoning_effort"]):
        summary_rows.append(dict(zip(["model_resolved", "representation", "prompting", "reasoning_effort"], keys)) | {"accuracy": accuracy_score(x.true, x.prediction), "macro_f1": f1_score(x.true, x.prediction, average="macro"), "n": len(x)})
    summary = pd.DataFrame(summary_rows); pred.to_csv(out / "llm_test_predictions.csv", index=False); summary.to_csv(out / "llm_results.csv", index=False)
    print(summary.to_string(index=False)); print(f"Completed: {len(good)}, failed calls: {len(failures)}")
    if aborted.is_set():
        remaining_now = sum((str(r.compound_id), m, rep, sn) not in done for m in a.models for rep in a.representations for sn, _ in conditions for _, r in test.iterrows())
        raise SystemExit(f"Run aborted on quota exhaustion with {remaining_now} evaluations still outstanding; rerun the same command after topping up.")

if __name__ == "__main__": main()
