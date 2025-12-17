"""Hash Embeddings for efficient representation learning.

Based on: Svenstrup, Hansen, Winther. "Hash embeddings for efficient word representations."
Advances in Neural Information Processing Systems. 2017.

Reference implementation: https://github.com/YannDubs/Hash-Embeddings
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Literal


class HashFamily:
    """Universal hash family as proposed by Carter and Wegman.

    Uses the formula: h_{a,b}(x) = ((ax + b) mod p) mod m
    where p > m and p is prime.

    Args:
        bins: Number of bins (buckets) to hash to.
        moduler: Prime number for temporary hashing. If None, a random prime > bins is used.
    """

    def __init__(self, bins: int, moduler: int | None = None):
        if moduler is not None and moduler <= bins:
            raise ValueError("moduler (p) should be > bins (m)")

        self.bins = bins
        self.moduler = moduler if moduler else self._next_prime(
            np.random.randint(self.bins + 1, 2**31 - 1)
        )

        # Track sampled parameters to avoid duplicates
        self.sampled_a: set[int] = set()
        self.sampled_b: set[int] = set()

    def _is_prime(self, x: int) -> bool:
        """Naive primality test."""
        if x < 2:
            return False
        for i in range(2, int(np.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True

    def _next_prime(self, n: int) -> int:
        """Get the next prime >= n."""
        while not self._is_prime(n):
            n += 1
        return n

    def draw_hash(self, a: int | None = None, b: int | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
        """Draw a single hash function from the family.

        Args:
            a: Optional fixed 'a' parameter.
            b: Optional fixed 'b' parameter.

        Returns:
            A hash function that maps torch.Tensor -> torch.Tensor.
        """
        if a is None:
            while a is None or a in self.sampled_a:
                a = np.random.randint(1, self.moduler - 1)
                if len(self.sampled_a) >= self.moduler - 2:
                    raise ValueError("Exhausted hash space, use a larger moduler")
            self.sampled_a.add(a)

        if b is None:
            while b is None or b in self.sampled_b:
                b = np.random.randint(0, self.moduler - 1)
                if len(self.sampled_b) >= self.moduler - 1:
                    raise ValueError("Exhausted hash space, use a larger moduler")
            self.sampled_b.add(b)

        # Capture a, b in closure
        _a, _b, _p, _m = a, b, self.moduler, self.bins
        return lambda x: ((_a * x + _b) % _p) % _m

    def draw_hashes(self, n: int) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        """Draw n hash functions from the family."""
        return [self.draw_hash() for _ in range(n)]


AggregationMode = Literal["sum", "concatenate", "median"]


class HashEmbedding(nn.Module):
    """Hash Embedding layer that uses multiple hashes to approximate embeddings.

    This is a parameter-efficient embedding that uses a shared pool of embedding
    vectors and multiple hash functions to map inputs to bucket indices.

    For each input x, the computation is:
        1. Hash x using k different hash functions to get k bucket indices
        2. Look up k embedding vectors from the shared pool
        3. Look up importance weights for x
        4. Aggregate weighted embeddings using sum/concatenate/median

    Args:
        num_embeddings: Number of unique input values (vocabulary size).
        embedding_dim: Size of each embedding vector.
        num_buckets: Size of the shared embedding pool. Should be << num_embeddings.
            If None, defaults to (num_embeddings * num_hashes) / embedding_dim.
        num_hashes: Number of hash functions to use (typically 2-3).
        aggregation_mode: How to aggregate the hashed embeddings.
            - "sum": Element-wise sum (default, same as mean with learned scaling)
            - "concatenate": Concatenate all k embeddings
            - "median": Element-wise median
        seed: Random seed for reproducibility.

    Attributes:
        shared_embeddings: The shared pool of embeddings [num_buckets, embedding_dim].
        importance_weights: Per-input importance weights [num_embeddings, num_hashes].
        output_dim: Effective output dimension after aggregation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_buckets: int | None = None,
        num_hashes: int = 2,
        aggregation_mode: AggregationMode = "sum",
        seed: int | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_hashes = num_hashes
        self.aggregation_mode = aggregation_mode
        self.seed = seed

        # Default num_buckets if not specified
        if num_buckets is None:
            num_buckets = max(1, (num_embeddings * num_hashes) // embedding_dim)
        self.num_buckets = num_buckets

        # Shared embedding pool
        self.shared_embeddings = nn.Embedding(
            num_embeddings=self.num_buckets,
            embedding_dim=self.embedding_dim,
        )

        # Importance weights for each input
        self.importance_weights = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.num_hashes,
        )

        # Set up aggregation function
        if aggregation_mode == "sum":
            self._aggregate = lambda x: torch.sum(x, dim=-1)
        elif aggregation_mode == "concatenate":
            self._aggregate = lambda x: torch.cat(
                [x[..., i] for i in range(self.num_hashes)], dim=-1
            )
        elif aggregation_mode == "median":
            self._aggregate = lambda x: torch.median(x, dim=-1)[0]
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")

        # Calculate output dimension
        if aggregation_mode == "concatenate":
            self.output_dim = self.embedding_dim * self.num_hashes
        else:
            self.output_dim = self.embedding_dim

        # Initialize hash functions
        if seed is not None:
            np.random.seed(seed)
        hash_family = HashFamily(self.num_buckets)
        self.hashes = hash_family.draw_hashes(self.num_hashes)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embedding weights."""
        nn.init.normal_(self.shared_embeddings.weight, std=0.1)
        nn.init.normal_(self.importance_weights.weight, std=0.0005)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input: LongTensor of shape [batch_size, ...] containing input indices.

        Returns:
            Tensor of shape [batch_size, ..., output_dim] containing embeddings.
        """
        # Map inputs to valid range for importance weights
        idx_importance = input % self.num_embeddings

        # Get bucket indices for each hash function
        # Stack results: [batch_size, ..., num_hashes]
        idx_buckets = torch.stack(
            [h(input) % self.num_buckets for h in self.hashes],
            dim=-1
        )

        # Look up shared embeddings for each hash
        # Result: [batch_size, ..., embedding_dim, num_hashes]
        shared_embeds = torch.stack(
            [self.shared_embeddings(idx_buckets[..., i]) for i in range(self.num_hashes)],
            dim=-1
        )

        # Look up importance weights
        # Result: [batch_size, ..., num_hashes] -> [batch_size, ..., 1, num_hashes]
        importance = self.importance_weights(idx_importance).unsqueeze(-2)

        # Weighted embeddings: [batch_size, ..., embedding_dim, num_hashes]
        weighted_embeds = importance * shared_embeds

        # Aggregate: [batch_size, ..., output_dim]
        output = self._aggregate(weighted_embeds)

        return output

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"num_buckets={self.num_buckets}, "
            f"num_hashes={self.num_hashes}, "
            f"aggregation_mode={self.aggregation_mode}, "
            f"output_dim={self.output_dim}"
        )
