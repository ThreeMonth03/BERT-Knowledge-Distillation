import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size: int = 30522, dim: int = 768, dropout: float = 0.1, max_position_embeddings: int = 512):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.word_embeddings(input_ids)
        return embeddings

if __name__ == '__main__':
    inputs = torch.randint(1000,(256,512))
    print(inputs.shape)
    emb = Embeddings()
    print(emb(inputs).shape)
