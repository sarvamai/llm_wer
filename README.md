## Background and Motivation

To evaluate Automatic Speech Recognition, we traditionally use metrics like Word Error Rate (WER) and Character Error Rate (CER). These metrics work by strictly comparing the ASR's transcribed text to a perfect, human-verified reference text, word for word.

However, human language, especially in a multilingual context like India, is fluid and filled with nuances. A rigid, word-for-word comparison often penalizes ASR models for making stylistic or formatting choices that are perfectly acceptable and do not change the meaning of the text. This leads to an inflated error rate and an inaccurate assessment of the ASR's true performance.

## Why Standard WER is Insufficient for Indic Languages

Standard WER can be misleading because it fails to understand context and semantic equivalence. It flags any textual difference as an error, even when no actual mistake in understanding has occurred.

Here are a few cases where the WER would be high, but the transcription is functionally correct:

- **Use of Different Scripts for Loanwords:** In Hindi, it is common to use English words. An ASR might write the English word "doctor" in its native Latin script, while the reference text might have it transliterated into Devanagari script ("डॉक्टर").
    - **Reference:** `वह डॉक्टर के पास गया`
    - **ASR Output:** `वह doctor के पास गया`
    - **Problem:** While both sentences are identical in meaning and would be understood perfectly by a Hindi speaker, standard WER would count "doctor" as an error, unfairly penalizing the model.
- **Multiple Valid Spellings:** Many words in Indic languages have several accepted spellings. For instance, in Tamil, names and even common words can be written in slightly different ways without any loss of meaning.
    - **Reference:** `அவர்கள் ஒன்றாக வேலை செய்கிறார்கள்` (Avargal oṉṟāka vēlai ceykiṟārkaḷ)
    - **ASR Output:** `அவுங்க ஒண்ணா வேலை செய்றாங்க` (Avuṅka oṇṇā vēlai ceyṟāṅka)
    - **Problem:** The ASR has produced a colloquial but perfectly valid and understandable version of the sentence. A human would perceive no error here, but a strict WER calculation would flag multiple words as incorrect, resulting in a very high error rate.

## Methodology: How LLM-WER Solves This

To address these shortcomings, we introduce LLM-WER, a metric that leverages the deep language understanding of a Large Language Model (LLM) to make a more intelligent evaluation.

The process consists of three high-level steps:

- **Identify Differences:** First, the system performs a standard comparison between the ASR output and the reference text to find the specific segments that do not match exactly.
- **Consult the Language Model (LLM)**: For each mismatched segment, we don't immediately count it as an error. Instead, we present both the reference segment and the ASR's version to a powerful LLM. We then ask the LLM to verify two critical properties: are the phrases **semantically equivalent** (do they mean the same thing?) and are they **phonetically similar** (do they sound alike when spoken?). The LLM leverages its deep understanding of context and pronunciation rules to determine if the core message is preserved and the sound is acceptably close, even with variations in spelling. The LLM is designed to default to non-equivalence in any ambiguous case, ensuring that a match is only confirmed when there is a high degree of certainty. This approach grounds the judgment, minimizing the risk of false positives.
- **Score Intelligently**: If the LLM confirms that the two segments are equivalent in both meaning and sound, we treat the ASR's output as correct, and the difference is not counted as an error. The final WER is then recalculated based only on the parts that the LLM identified as genuine mistakes.

## Conclusion

By integrating the semantic judgment of an LLM, the LLM-WER metric provides a far more accurate and fair assessment of an ASR model's performance. It moves beyond rigid, literal comparisons and instead measures what truly matters: whether the ASR correctly understood the meaning of what was spoken. This approach allows us to appreciate the stylistic flexibility of Indic languages and build ASR systems that are better aligned with how humans naturally speak and write.
