"""Token-aware chunks retain original text using tokenizer offsets."""

import hashlib

from policypilot.models import Filing, Passage


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def chunks(self, text: str, filing: Filing, overlap: int = 32) -> list[Passage]:
        tokenizer = self.model.tokenizer
        limit = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
        if not 0 <= overlap < limit:
            raise ValueError("Invalid token overlap")
        offsets = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )["offset_mapping"]
        chunks = []
        for start in range(0, len(offsets), limit - overlap):
            end = min(start + limit, len(offsets))
            part = text[offsets[start][0] : offsets[end - 1][1]].strip()
            if not part:
                continue
            # Retokenizing a slice can alter word-piece boundaries; enforce the true limit.
            while (
                len(tokenizer(part, add_special_tokens=True)["input_ids"])
                > self.model.max_seq_length
            ):
                end -= 1
                part = text[offsets[start][0] : offsets[end - 1][1]].strip()
            identifier = hashlib.sha256(
                f"{filing.ticker}|{filing.accession}|{start}|{part}".encode()
            ).hexdigest()
            chunks.append(
                Passage(
                    **filing.model_dump(), chunk_id=identifier, text=part, chunk_index=len(chunks)
                )
            )
            if end == len(offsets):
                break
        return chunks
