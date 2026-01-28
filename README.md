# Finnish News Summarizer (Llama 3.2)

A lightweight, fine-tuned LLM designed to summarize and simplify Finnish news articles into "Selkosuomi" (Plain Finnish).

Built with [Unsloth](https://github.com/unslothai/unsloth) and fine-tuned on the **Llama-3.2-3B-Instruct** architecture, this model runs efficiently on consumer hardware (8GB VRAM).

Dataset: Yle News 2014-2020 [Selkosuomi Parallel Corpus](https://vlo.clarin.eu/record/urn_58_nbn_58_fi_58_lb-2024011703?1).

## Features

- **Task:** Summarization & Text Simplification.
- **Language:** Finnish (Suomi) -> Plain Finnish (Selkosuomi).
- **Architecture:** Llama-3.2-3B (Quantized 4-bit).
- **Performance:** ~2x faster inference using Unsloth's native optimizations.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/yourusername/finnish-news-summarizer-llama-3.2](https://github.com/yourusername/finnish-news-summarizer-llama-3.2)
   cd finnish-news-summarizer-llama-3.2
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Data:**

   Dataset download: [link](https://www.kielipankki.fi/download/YLE/fi-selko-par/2014-2020-selko-par-sent-src/).

   Note on Access: The dataset is not included in this repository due to licensing restrictions. It requires a university or institutional login to access.

4. **Repo Structure:**
   ```
   ├── train_checkpoints/ # (Ignored) Stores trained adapters
   ├── summarization_dataset/ # (Ignored) Cached HuggingFace dataset
   ├── ylenews-fi-.../ # (Ignored) Place your downloaded CSVs here!
   ├── summarizer_finetuning.ipynb # Full training notebook
   └── requirements.txt # Python dependencies
   ```

## Quick Start

You can run the model using the provided Python script. Note that you must use the specific instruction format used during training ("Tiivistä seuraava uutinen selkosuomeksi").

    ```python
    from unsloth import FastLanguageModel
    import torch

    # Try new model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "content/train_checkpoints/checkpoint-120",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    article = """(Paste your Finnish news text here)"""

    # same format as training
    messages = [
        {
            "role": "user",
            # MUST match the training formatting perfectly!
            "content": f"Tiivistä seuraava uutinen selkosuomeksi:\n\n{test_article}\n"
        }
    ]

    # tokenize with the "Generation Prompt"
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # so model knows to start answering
        return_tensors = "pt",
    ).to("cuda")

    # tells the model that every single token is important and not to ignore anything
    attention_mask = torch.ones_like(inputs)

    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = 512,
        use_cache = True,
        temperature = 0.4, # you can try higher values for more creative output ;)
    )

    response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]

    print("--- SUMMARY ---")
    print(response)
    ```

## License

- **Model:** Llama 3.2 Community License.

- **Dataset:** Yle News / Kielipankki (DATASET_LICENSE.txt).

- **Code:** MIT License.

## Other

Read the report ('Report.pdf') for more information.
