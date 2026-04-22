import argparse
from main import process_dataset_with_predictions


def main():
    parser = argparse.ArgumentParser(
        prog="llm-wer",
        description="Evaluate ASR transcriptions using LLM-assisted Word Error Rate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", required=True, help="Path to CSV or JSONL dataset file.")
    parser.add_argument("--ref-col", required=True, help="Column name for reference transcriptions.")
    parser.add_argument("--pred-col", required=True, help="Column name for ASR predictions.")
    parser.add_argument("--audio-col", required=True, help="Column name for audio file paths.")
    parser.add_argument("--lang-col", default="language", help="Column name for language labels.")

    # LLM
    llm = parser.add_argument_group("LLM configuration")
    llm.add_argument("--model", default="google/gemini-2.5-flash", help="Model name.")
    llm.add_argument("--no-gemini", dest="gemini", action="store_false",
                     help="Use an OpenAI-compatible API instead of Gemini via Vertex AI.")
    llm.add_argument("--creds-path", default=None, help="GCP service account JSON path (required for Gemini).")
    llm.add_argument("--gemini-location", default="us-central1", help="Vertex AI region.")
    llm.add_argument("--api-key", default="", help="API key for OpenAI-compatible endpoints.")
    llm.add_argument("--base-url", default="", help="Base URL for OpenAI-compatible endpoints.")

    # Sheets (optional)
    sheets = parser.add_argument_group("Google Sheets output (optional)")
    sheets.add_argument("--sheet-name", default=None, help="Spreadsheet name to push results to.")
    sheets.add_argument("--worksheet-name", default=None, help="Worksheet (tab) name inside the spreadsheet.")

    parser.add_argument("--ignore-cache", action="store_true", help="Ignore cached LLM responses and re-query.")

    args = parser.parse_args()

    process_dataset_with_predictions(
        dataset_path=args.dataset,
        reference_col_name=args.ref_col,
        predicted_col_name=args.pred_col,
        audio_filepath_col_name=args.audio_col,
        language_col_name=args.lang_col,
        model_name=args.model,
        gemini=args.gemini,
        creds_path=args.creds_path,
        gemini_location=args.gemini_location,
        api_key=args.api_key,
        base_url=args.base_url,
        output_sheet_name=args.sheet_name,
        output_worksheet_name=args.worksheet_name,
        ignore_cache=args.ignore_cache,
    )


if __name__ == "__main__":
    main()
