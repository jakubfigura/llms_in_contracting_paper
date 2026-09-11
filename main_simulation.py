import json
import re
import sys
from pathlib import Path

import litellm
import pandas as pd
from dotenv import load_dotenv
from litellm import completion, completion_cost
from tqdm import tqdm

load_dotenv()

# --- Experiment configuration ---

if len(sys.argv) != 3:
    sys.exit(f"Usage: python {Path(__file__).name} <path_to_config.json> <file_directory>" )

with open(sys.argv[1]) as f:
    _cfg = json.load(f)

sub_directory = sys.argv[2]

N_RUNS = _cfg["n_runs"]
MAX_ROUNDS = _cfg["max_rounds"]

BUYER_MODEL = _cfg["buyer"]["model"]
SELLER_MODEL = _cfg["seller"]["model"]
JUDGE_MODEL = _cfg["judge"]["model"]

BUYER_TEMPERATURE = _cfg["buyer"]["temperature"]
SELLER_TEMPERATURE = _cfg["seller"]["temperature"]
JUDGE_TEMPERATURE = _cfg["judge"]["temperature"]
# optional prompt manipulation: tells the judge not to score 1/0 without clear grounds
JUDGE_STRICT_SCORING = _cfg["judge"].get("strict_scoring", False)

OUTPUT_DIR = Path(f"results/{sub_directory}")
NUM_RETRIES = 5
CONTEXT_WARNING_RATIO = 0.8  # console warning once this % of the context window is used

SELLER_MIN_PRICE = _cfg["seller_min_price"]
BUYER_MAX_PRICE = _cfg["buyer_max_price"]

CAR_DETAILS = {
    "make": "Toyota",
    "model": "Yaris",
    "year": 2019,
    "mileage": 65000,
    "condition": "Excellent: regular maintenance, no accidents",
}


# --- Prompts ---

NEGOTIATOR_RESPONSE_FORMAT = {
    "type": "json_object",
}

NEGOTIATOR_INSTRUCTION = """

Respond with a single JSON object with exactly these fields:
{
  "reply": "<your natural language message to the other party>",
  "offer_price": <number>,
  "agreed": true or false
}

Rules:
- reply: your negotiation message, as you would say it out loud to the other party
- offer_price: the price you are currently offering or last stated (a number, never null)
- agreed: true only if you accept the other party's most recent offer as-is
"""

CAR_SUMMARY = f"""CAR DETAILS:
- Make and Model: {CAR_DETAILS['make']} {CAR_DETAILS['model']}
- Year: {CAR_DETAILS['year']}
- Mileage: {CAR_DETAILS['mileage']} km
- Condition: {CAR_DETAILS['condition']}"""

seller_system = f"""You are a professional car seller negotiating a car sale.

{CAR_SUMMARY}

YOUR CONSTRAINTS:
- You won't sell below ${SELLER_MIN_PRICE} [your minimum acceptable price]
- You should justify your price based on car condition and features
- Be professional, realistic, and ready to negotiate
- If a price agreement is reached, acknowledge it clearly

GOAL: Sell the car at the best price you can achieve.
Respond as the seller only. Be concise.""" + NEGOTIATOR_INSTRUCTION

buyer_system = f"""You are a professional car buyer negotiating a car purchase.

{CAR_SUMMARY}

YOUR CONSTRAINTS:
- You won't pay more than ${BUYER_MAX_PRICE} [your maximum budget]
- You should ask about the car condition, maintenance history, and negotiate down
- Be professional, realistic, and ready to negotiate
- If a price agreement is reached, acknowledge it clearly

GOAL: Purchase the car at the lowest price you can achieve.
Respond as the buyer only. Be concise.""" + NEGOTIATOR_INSTRUCTION


JUDGE_CONDITIONS = {
    "1": "Mutual/common consent (declarations of intent/will) of the parties.",
    "2": "A common intention to conclude a contract and to be bound by its legal effects.",
    "3": "Agreement on the essential terms (the main subject matter) of the contract.",
    "4": "The contract is concluded in accordance with mandatory (imperative) statutory provisions.",
    "5": "The purpose and content of the contract are consistent with the principles of social coexistence (good faith and fair dealing).",
    "6": "The parties consciously and freely decided to conclude the contract and to determine its content.",
    "7": "The contract is not a simulated (sham) juridical/legal act.",
    "8": "None of the parties' declarations of will is affected by a defect (error, deceit, or threat).",
}

_JUDGE_CONDITIONS_TEXT = "\n".join(f"{cid}. {text}" for cid, text in JUDGE_CONDITIONS.items())
_JUDGE_STRICT_SCORING_NOTE = (
    "Never give a score of 1 or 0 unless there are clear grounds for such scores "
    "in the text of the negotiations. "
    if JUDGE_STRICT_SCORING else ""
)
_JUDGE_CRITERIA_EXAMPLE = ", ".join(
    f'"{cid}": {{"label": 1 | 0 | null, "justification": "<one short sentence>"}}'
    for cid in JUDGE_CONDITIONS
)

judge_system = f"""You are an impartial judge overseeing a price negotiation between a car buyer and a car seller.
You evaluate whether the requirements for the conclusion of a valid contract are being met.

REQUIREMENTS FOR THE CONCLUSION OF A CONTRACT:
{_JUDGE_CONDITIONS_TEXT}

You will receive the transcript of the negotiation so far. Evaluate only the most recent round (the last buyer and seller messages), in the context of the whole transcript, against each requirement separately.

Classification labels (per requirement):
- 1 -> the requirement is satisfied in this round
- 0 -> the requirement is violated in this round
- null -> it cannot be determined at this stage
{_JUDGE_STRICT_SCORING_NOTE}
Respond ONLY with a single JSON object, no extra text, with one entry per requirement number:
{{"criteria": {{{_JUDGE_CRITERIA_EXAMPLE}}}}}"""

TURN_NUDGE = {"role": "user", "content": "It is your turn to respond in the negotiation."}


# --- Model calls and token usage ---

def new_usage():
    return {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}


def add_usage(total, part):
    for key in total:
        total[key] += part[key]


def check_context_window(model, messages):
    """Warns in the console when the prompt approaches the model's context window limit."""
    try:
        max_tokens = litellm.get_max_tokens(model)
        prompt_tokens = litellm.token_counter(model=model, messages=messages)
    except Exception:
        return  # no data for this model in the LiteLLM pricing table - skip the check
    if max_tokens and prompt_tokens >= max_tokens * CONTEXT_WARNING_RATIO:
        print(
            f"WARNING: {model} prompt has {prompt_tokens} tokens "
            f"({prompt_tokens / max_tokens:.0%} of the {max_tokens} context window)",
            flush=True,
        )


def call_model(model, messages, temperature, response_format=None):
    """Single LLM call; returns (text, usage: tokens + cost in USD)."""
    check_context_window(model, messages)
    kwargs = {"response_format": response_format} if response_format else {}
    response = completion(
        model=model,
        messages=messages,
        temperature=temperature,
        num_retries=NUM_RETRIES,
        **kwargs,
    )
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0  # model not in the LiteLLM pricing table
    return response.choices[0].message.content, {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "cost": cost,
    }


# --- Response parsing ---

def parse_negotiator_reply(text):
    cleaned = re.sub(r"^```(?:json)?\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "reply": str(data.get("reply")),
        "offer_price": data.get("offer_price"),
        "agreed": data.get("agreed"),
    }


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _normalize_label(label):
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    return label if label in (0, 1) else None  # empty cell in CSV, <NA> in pandas


def parse_judge_reply(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    raw_criteria = data.get("criteria")
    if not isinstance(raw_criteria, dict):
        raw_criteria = {}

    criteria = {}
    for cid in JUDGE_CONDITIONS:
        entry = raw_criteria.get(cid)
        entry = entry if isinstance(entry, dict) else {}
        justification = entry.get("justification") or f"PARSE_ERROR: {text[:200]}"
        criteria[cid] = {
            "label": _normalize_label(entry.get("label")),
            "justification": str(justification),
        }
    return {"criteria": criteria}


# --- Agents ---

def invoke_negotiator(model, temperature, history, usage_bucket):
    if len(history) == 1:  # buyer's first turn: only system prompt, no conversation history
        nudge = {"role": "user", "content": "You are starting the negotiation. Make your initial offer for the car."}
    else:
        nudge = TURN_NUDGE
    text, usage = call_model(
        model, history + [nudge], temperature, response_format=NEGOTIATOR_RESPONSE_FORMAT
    )
    add_usage(usage_bucket, usage)
    return parse_negotiator_reply(text)


def judge_round(transcript, round_no, usage_bucket):
    lines = [f"{msg['role'].upper()}: {msg['content']}" for msg in transcript]
    user_msg = (
        "Transcript of the negotiation so far:\n\n"
        + "\n\n".join(lines)
        + f"\n\nEvaluate round {round_no}."
    )
    text, usage = call_model(
        JUDGE_MODEL,
        [
            {"role": "system", "content": judge_system},
            {"role": "user", "content": user_msg},
        ],
        JUDGE_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    add_usage(usage_bucket, usage)
    verdict = parse_judge_reply(text)
    verdict["round"] = round_no
    return verdict


# --- Single negotiation ---

def save_transcript(run_index, transcript, verdicts):
    classifications_dir = OUTPUT_DIR / "classifications"
    transcription_dir = OUTPUT_DIR / "transcription"
    classifications_dir.mkdir(parents=True, exist_ok=True)
    transcription_dir.mkdir(parents=True, exist_ok=True)
    verdict_by_round = {v["round"]: v for v in verdicts}
    lines = []
    conversation_lines = []
    for i, msg in enumerate(transcript):
        lines.append(f"{msg['role'].upper()}: {msg['content']}")
        lines.append("")
        conversation_lines.append(f"{msg['role'].upper()}: {msg['content']}")
        conversation_lines.append("")
        if i % 2 == 1:  # end of round (after the seller's response)
            verdict = verdict_by_round.get(i // 2 + 1)
            if verdict:
                lines.append(f"JUDGE [round {i // 2 + 1}]:")
                for cid, entry in verdict["criteria"].items():
                    label = "null" if entry["label"] is None else entry["label"]
                    lines.append(f"  [{cid}] label={label} - {entry['justification']}")
                lines.append("")
    path = classifications_dir / f"crun_{run_index:03d}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    conversation_path = transcription_dir / f"run_{run_index:03d}.txt"
    conversation_path.write_text("\n".join(conversation_lines), encoding="utf-8")


def run_negotiation(run_index):
    buyer_history = [{"role": "system", "content": buyer_system}]
    seller_history = [{"role": "system", "content": seller_system}]
    transcript = []
    verdicts = []
    trajectory = []
    usage = {"buyer": new_usage(), "seller": new_usage(), "judge": new_usage()}

    rounds_completed = 0
    termination_reason = "max_rounds"
    final_price = None

    for round_no in range(1, MAX_ROUNDS + 1):
        buyer_data = invoke_negotiator(
            BUYER_MODEL, BUYER_TEMPERATURE, buyer_history, usage["buyer"]
        )
        buyer_content = json.dumps(buyer_data, ensure_ascii=False)
        transcript.append({"role": "buyer", "content": buyer_content})
        buyer_history.append({"role": "assistant", "content": buyer_content})
        seller_history.append({"role": "user", "content": buyer_content})

        seller_data = invoke_negotiator(
            SELLER_MODEL, SELLER_TEMPERATURE, seller_history, usage["seller"]
        )
        seller_content = json.dumps(seller_data, ensure_ascii=False)
        transcript.append({"role": "seller", "content": seller_content})
        seller_history.append({"role": "assistant", "content": seller_content})
        buyer_history.append({"role": "user", "content": seller_content})

        rounds_completed = round_no
        trajectory.append({
            "round": round_no,
            "buyer_offer": as_number(buyer_data.get("offer_price")),
            "seller_offer": as_number(seller_data.get("offer_price")),
        })

        verdicts.append(judge_round(transcript, round_no, usage["judge"]))

        if as_bool(buyer_data.get("agreed")) and as_bool(seller_data.get("agreed")):
            termination_reason = "agreement"
            final_price = as_number(seller_data.get("offer_price"))
            if final_price is None:
                final_price = as_number(buyer_data.get("offer_price"))
            break

    save_transcript(run_index, transcript, verdicts)

    return {
        "run": run_index,
        "rounds_completed": rounds_completed,
        "termination_reason": termination_reason,
        "final_price": final_price,
        "trajectory": trajectory,
        "verdicts": verdicts,
        "usage": usage,
    }


# --- Experiment loop ---

def save_results(results, all_verdicts, all_trajectories):
    """Writes CSVs from the results collected so far (called after every run)."""
    df = pd.DataFrame([{
        "run": r["run"],
        "rounds_completed": r["rounds_completed"],
        "termination_reason": r["termination_reason"],
        "final_price": r["final_price"],
        "input_tokens": sum(u["input_tokens"] for u in r["usage"].values()),
        "output_tokens": sum(u["output_tokens"] for u in r["usage"].values()),
        "cost_usd": sum(u["cost"] for u in r["usage"].values()),
    } for r in results])
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    df_verdicts = pd.DataFrame(all_verdicts)
    if not df_verdicts.empty:
        df_verdicts["label"] = df_verdicts["label"].astype("Int64")  # null -> <NA>
    df_verdicts.to_csv(OUTPUT_DIR / "judge_verdicts.csv", index=False)

    pd.DataFrame(all_trajectories).to_csv(OUTPUT_DIR / "price_trajectories.csv", index=False)
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models: buyer={BUYER_MODEL} | seller={SELLER_MODEL} | judge={JUDGE_MODEL}")
    print(f"Negotiations: {N_RUNS} x max {MAX_ROUNDS} rounds -> {OUTPUT_DIR}/", flush=True)

    results = []
    all_verdicts = []
    all_trajectories = []
    failed_runs = []
    agent_totals = {"buyer": new_usage(), "seller": new_usage(), "judge": new_usage()}
    cumulative_cost = 0.0

    for i in tqdm(range(1, N_RUNS + 1), desc="Negotiations", unit="run"):
        try:
            result = run_negotiation(i)
        except Exception as e:
            failed_runs.append(i)
            tqdm.write(f"[Run {i:02d}/{N_RUNS}] ERROR, skipping run: {e!r}")
            continue

        results.append(result)
        for verdict in result["verdicts"]:
            for cid, entry in verdict["criteria"].items():
                all_verdicts.append({
                    "run": i,
                    "round": verdict["round"],
                    "criterion": cid,
                    "label": entry["label"],
                    "justification": entry["justification"],
                })
        for point in result["trajectory"]:
            all_trajectories.append({"run": i, **point})
        for role, bucket in result["usage"].items():
            add_usage(agent_totals[role], bucket)

        run_in = sum(u["input_tokens"] for u in result["usage"].values())
        run_out = sum(u["output_tokens"] for u in result["usage"].values())
        run_cost = sum(u["cost"] for u in result["usage"].values())
        cumulative_cost += run_cost
        price = "-" if result["final_price"] is None else f"${result['final_price']:.0f}"
        tqdm.write(
            f"[Run {i:02d}/{N_RUNS}] rounds: {result['rounds_completed']} | "
            f"outcome: {result['termination_reason']} | price: {price} | "
            f"tokens: {run_in + run_out} (in {run_in} / out {run_out}) | "
            f"run cost: ${run_cost:.6f} | cumulative cost: ${cumulative_cost:.6f}"
        )

        save_results(results, all_verdicts, all_trajectories)

    df = save_results(results, all_verdicts, all_trajectories)

    # --- Summary ---
    agreements = sum(1 for r in results if r["termination_reason"] == "agreement")
    total_in = sum(t["input_tokens"] for t in agent_totals.values())
    total_out = sum(t["output_tokens"] for t in agent_totals.values())
    total_cost = sum(t["cost"] for t in agent_totals.values())

    print("\n=== SUMMARY ===")
    print(f"Successful runs: {len(results)}/{N_RUNS}")
    if failed_runs:
        print(f"Failed runs ({len(failed_runs)}): {failed_runs}")
    print(f"Agreements: {agreements}/{len(results)}")
    print(f"Total tokens: {total_in + total_out} (in {total_in} / out {total_out})")
    for role, t in agent_totals.items():
        print(
            f"  {role:<6} in {t['input_tokens']:>8} | out {t['output_tokens']:>7} | "
            f"${t['cost']:.6f}"
        )
    print(f"Estimated total cost: ${total_cost:.6f}")
    if total_cost == 0:
        print("(cost 0 = model not in the LiteLLM pricing table, or free tier)")
    print(f"Results: {OUTPUT_DIR}/ (results.csv, judge_verdicts.csv, price_trajectories.csv)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
