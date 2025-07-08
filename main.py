from pydantic import BaseModel
from llm_api import ChatCompletionsAPI
from difflib import SequenceMatcher
import logging
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
from utilities import wer, cer, IndicNormalizer, push_to_sheet

logger = logging.getLogger(__name__)

try:
    PROMPT_TEMPLATE = (Path(__file__).parent / "prompt_template.txt").read_text()
except FileNotFoundError:
    raise FileNotFoundError("prompt_template.txt not found. Please create it in the same directory as main.py.")

class LLMResponse(BaseModel):
    index: int
    equivalent: bool
    reasoning: str

def get_segments(reference_string: str, predicted_string: str, key: Any) -> List[Dict[str, Any]]:
    try:
        reference_words = reference_string.strip().split()
        predicted_words = predicted_string.strip().split()
        if not reference_words and not predicted_words:
            return []
        
        matcher = SequenceMatcher(None, reference_words, predicted_words)
        return [
            {
                "reference": " ".join(reference_words[i1:i2]),
                "prediction": " ".join(predicted_words[j1:j2]),
                "tag": tag,
                "key": key,
                "r_start": i1, "r_end": i2,
                "p_start": j1, "p_end": j2,
                "segment_idx": segment_idx,
            }
            for segment_idx, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes())
        ]
    except Exception as e:
        logger.error(f"Error in get_segments for key {key}: {e}")
        return []

def build_prompt(segment_pair: Dict[str, str]) -> str:
    prompt = PROMPT_TEMPLATE + "\n\n**INPUT:**\n"
    json_objects = [{"index": 0, "reference": segment_pair["reference"], "prediction": segment_pair["prediction"]}]
    prompt += json.dumps(json_objects, indent=2, ensure_ascii=False)
    return prompt

def load_and_validate_dataset(
    dataset_path: str, required_cols: set
) -> pd.DataFrame:
    path_obj = Path(dataset_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{dataset_path} does not exist")

    if path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(path_obj)
    elif path_obj.suffix.lower() in {".jsonl", ".json"}:
        df = pd.read_json(path_obj, lines=True)
    else:
        raise ValueError("Unsupported dataset format. Only CSV or JSONL accepted.")

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {', '.join(missing_cols)}")
    
    return df

def normalize_and_calculate_original_error_rates(df: pd.DataFrame, ref_col: str, pred_col: str, lang_col: str) -> pd.DataFrame:
    normalizer = IndicNormalizer()
    df["norm_reference"] = normalizer.normalize_texts(df[ref_col].astype(str).to_list(), df[lang_col].astype(str).to_list())
    df["norm_prediction"] = normalizer.normalize_texts(df[pred_col].astype(str).to_list(), df[lang_col].astype(str).to_list())

    df["original_wer"] = [wer(ref, pred) for ref, pred in zip(df["norm_reference"], df["norm_prediction"])]
    df["original_cer"] = [cer(ref, pred) for ref, pred in zip(df["norm_reference"], df["norm_prediction"])]
    
    return df

def extract_unique_segments(df: pd.DataFrame) -> Tuple[Dict[int, List[Dict]], Dict[Tuple[str, str], List[Dict]]]:
    row_segment_map = {}
    unique_segments_to_process = {}

    for idx, row in df.iterrows():
        segments = get_segments(row["norm_reference"], row["norm_prediction"], key=idx) # type: ignore
        row_segment_map[idx] = segments
        for segment in segments:
            if segment["tag"] != "equal" and segment["reference"].strip() and segment["prediction"].strip():
                segment_pair = (segment["reference"], segment["prediction"])
                if segment_pair not in unique_segments_to_process:
                    unique_segments_to_process[segment_pair] = []
                unique_segments_to_process[segment_pair].append({"row_idx": idx, "segment_idx": segment["segment_idx"]})
    
    return row_segment_map, unique_segments_to_process

def query_llm_for_equivalence(
    unique_segments: Dict[Tuple[str, str], List[Dict]],
    dataset_name: str,
    api: ChatCompletionsAPI,
    ignore_cache: bool = False,
) -> Tuple[list, list]:
    cache_dir = Path(__file__).parent / "outputs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset_name}_cache.jsonl"

    if ignore_cache and cache_file.exists():
        cache_file.unlink()

    cached_responses_map = {}
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    cached_item = json.loads(line)
                    key = cached_item.get("key")
                    if key and "reference" in key and "prediction" in key:
                        cached_responses_map[(key["reference"], key["prediction"])] = cached_item
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed cache line in {cache_file}: '{line.strip()}'. Error: {e}")

    segments_to_query = {
        (ref, pred): occurrences
        for (ref, pred), occurrences in unique_segments.items()
        if (ref, pred) not in cached_responses_map
    }
    
    successful_from_cache = [
        item for (ref, pred), item in cached_responses_map.items()
        if (ref, pred) in unique_segments
    ]

    newly_successful = []
    failed = []

    if segments_to_query:
        print(f"Found {len(successful_from_cache)} cached responses. Querying LLM for {len(segments_to_query)} new unique segments.")
        for ref, pred in segments_to_query.keys():
            segment_key = {"reference": ref, "prediction": pred}
            prompt = build_prompt(segment_key)
            api.append_to_request_queue(prompt=prompt, key=segment_key, schema=LLMResponse)
        
        newly_successful, failed = api.generate_responses_from_queue(output_file_path=cache_file)
    else:
        print(f"All {len(successful_from_cache)} required segments found in cache. No new API calls needed.")
    all_successful = successful_from_cache + newly_successful
    return all_successful, failed

def process_llm_responses(successful_responses: list, unique_segments_to_process: dict) -> Tuple[Dict[Tuple[int, int], bool], List[Dict]]:
    llm_verdicts = {}
    log_records = []
    
    for item in successful_responses:
        segment_info = item.get("key", {})
        llm_res = item.get("response", {})
        if not (segment_info and llm_res and "reference" in segment_info and "prediction" in segment_info):
            continue
        
        ref, pred = segment_info["reference"], segment_info["prediction"]
        is_equivalent = llm_res.get("equivalent", False)
        llm_verdicts[(ref, pred)] = is_equivalent
        log_records.append({
            "reference": ref,
            "prediction": pred,
            "equivalent": is_equivalent,
            "reasoning": llm_res.get("reasoning"),
        })

    equivalent_flags = {}
    for (ref, pred), occurrences in unique_segments_to_process.items():
        if llm_verdicts.get((ref, pred), False):
            for occ in occurrences:
                equivalent_flags[(occ["row_idx"], occ["segment_idx"])] = True
    
    return equivalent_flags, log_records

def reconstruct_and_score(df: pd.DataFrame, row_segment_map: dict, equivalent_flags: dict) -> pd.DataFrame:
    corrected_predictions = []
    corrected_references = []

    for idx, row in df.iterrows():
        segments = row_segment_map.get(idx, [])
        reconstructed_pred_parts = []
        reconstructed_ref_parts = []
        
        for segment in segments:
            is_equivalent = equivalent_flags.get((idx, segment["segment_idx"]), False)
            if segment["tag"] == "equal" or is_equivalent:
                reconstructed_pred_parts.append(segment["reference"])
                reconstructed_ref_parts.append(segment["reference"])
            else:
                reconstructed_pred_parts.append(segment["prediction"])
                reconstructed_ref_parts.append(segment["reference"])
                
        corrected_predictions.append(" ".join(reconstructed_pred_parts))
        corrected_references.append(" ".join(reconstructed_ref_parts))

    df["corrected_prediction"] = corrected_predictions
    df["corrected_reference"] = corrected_references
    df["corrected_wer"] = [wer(ref, pred) for ref, pred in zip(df["corrected_reference"], df["corrected_prediction"])]
    df["corrected_cer"] = [cer(ref, pred) for ref, pred in zip(df["corrected_reference"], df["corrected_prediction"])]
    
    return df

def save_outputs(df: pd.DataFrame, logs: List[Dict], failed: List[Dict], outputs_dir: Path, sheet_name: str, worksheet_prefix: str, creds_path: Path):
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(outputs_dir / "llm_wer.csv", index=False)
    print(f"Output saved to {outputs_dir / 'llm_wer.csv'}")
    push_to_sheet(df, sheet_name, f"{worksheet_prefix} - full output", creds_path)

    if logs:
        logs_df = pd.DataFrame(logs)
        logs_df.to_csv(outputs_dir / "llm_logs.csv", index=False)
        print(f"LLM logs saved to {outputs_dir / 'llm_logs.csv'}")
        push_to_sheet(logs_df, sheet_name, f"{worksheet_prefix} - llm logs", creds_path)

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(outputs_dir / "llm_failed_requests.csv", index=False)
        print(f"Failed LLM requests saved to {outputs_dir / 'llm_failed_requests.csv'}")


def process_dataset_with_predictions(
    dataset_path: str, 
    reference_col_name: str, 
    predicted_col_name: str, 
    audio_filepath_col_name: str, 
    creds_path: str,
    language_col_name: str = "language",
    output_sheet_name: str = "llm-wer-analysis",
    output_worksheet_name: str = "llm-wer-output",
    ignore_cache: bool = False,
    gemini_location: str = "us-central1",
):
    creds_path_obj = Path(creds_path)
    if not creds_path_obj.exists():
        raise FileNotFoundError(f"Credentials file not found at {creds_path}")

    api = ChatCompletionsAPI(
        model_name="google/gemini-2.5-flash",
        api_key="",
        base_url="",
        gemini=True,
        max_retries=0,
        timeout=None,
        creds_path=creds_path,
        location=gemini_location,
    )

    # api = ChatCompletionsAPI(
    #     model_name="sarvam-m",
    #     api_key="SARVAM-M-API-KEY",
    #     base_url="https://api.sarvam.ai/v1/",
    #     max_retries=0
    # )
    
    required_cols = {reference_col_name, predicted_col_name, audio_filepath_col_name, language_col_name}
    df = load_and_validate_dataset(dataset_path, required_cols)
    df = normalize_and_calculate_original_error_rates(df, reference_col_name, predicted_col_name, language_col_name)
    row_segment_map, unique_segments = extract_unique_segments(df)
    dataset_name = Path(dataset_path).stem
    successful, failed = query_llm_for_equivalence(
        unique_segments, dataset_name, api, ignore_cache=ignore_cache
    )
    equivalent_flags, log_records = process_llm_responses(successful, unique_segments)
    df = reconstruct_and_score(df, row_segment_map, equivalent_flags)

    print(f"Original WER mean: {df['original_wer'].mean():.4f}")
    print(f"Original CER mean: {df['original_cer'].mean():.4f}")
    print(f"Corrected WER mean: {df['corrected_wer'].mean():.4f}")
    print(f"Corrected CER mean: {df['corrected_cer'].mean():.4f}")
    
    outputs_dir = Path(__file__).parent / "outputs" / Path(dataset_path).stem
    outputs_dir.mkdir(parents=True, exist_ok=True)
    save_outputs(df, log_records, failed, outputs_dir, output_sheet_name, output_worksheet_name, creds_path_obj)

if __name__ == "__main__":
    process_dataset_with_predictions(
        dataset_path = "/path/to/path_with_predictions.csv", 
        reference_col_name="transcription", 
        predicted_col_name="prediction", 
        audio_filepath_col_name="audio_filepath",
        creds_path="/path/to/creds.json",
        language_col_name="language",
        output_sheet_name="LLM WER Analysis",
        output_worksheet_name="<name of worksheet>",
        ignore_cache=True,
    ) 