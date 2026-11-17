"""Hugging Face generation with shared sampling settings and token counting."""

import time

from .modeling import load_model, load_tokenizer


class HFBackend:
    def __init__(self, config, model=None, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(config)
        self.model = model if model is not None else load_model(config)
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = [eos] if isinstance(eos, int) else list(eos)

    def resolved_generation_config(self, mode):
        from transformers import GenerationConfig

        # Start from model defaults, explicitly neutralize unwanted penalties/constraints.
        settings = self.model.generation_config.to_dict()
        settings.update(self.config["generation"][mode])
        settings.update(
            do_sample=True,
            max_new_tokens=self.config["generation"]["max_new_tokens"],
            repetition_penalty=1.0,
            use_cache=True,
            num_beams=1,
            num_return_sequences=1,
            min_length=0,
            min_new_tokens=0,
            no_repeat_ngram_size=0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.stop_ids,
        )
        return GenerationConfig.from_dict(settings)

    def generate_one(self, prompt_ids, mode, seed):
        import torch
        from transformers import set_seed

        set_seed(seed)
        if (
            len(prompt_ids) + self.config["generation"]["max_new_tokens"]
            > self.model.config.max_position_embeddings
        ):
            raise ValueError("Prompt plus generation cap exceeds model context")
        ids = torch.tensor([prompt_ids], device=self.model.device)
        torch.cuda.synchronize(self.model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                generation_config=self.resolved_generation_config(mode),
            )
        torch.cuda.synchronize(self.model.device)
        return {
            "token_ids": output[0, len(prompt_ids) :].tolist(),
            "elapsed_seconds": time.perf_counter() - start,
            "batch_size": 1,
        }

    def generate_many(self, requests):
        return [self.generate_one(**request) for request in requests]


def make_backend(config):
    return HFBackend(config)
