import flax.linen as nn
import jax.numpy as jnp

from typing import Any, Tuple

from agent import network


class DefenderBuilder(nn.Module):
    embed_dim: int

    map_size: Tuple[int, int]

    @nn.compact
    def __call__(self, tower, scalar):
        """

        Args:
            tower_attr: ([batch, length, attr], [batch, length, 2])
            scalar: [batch, feature]
        """

        tower_attr, tower_coord = tower

        # embed tower attribute using Embedder
        embedding = network.Embedder(embed_dim=self.embed_dim)(tower_attr)

        # apply Transformer to embedding
        cls, encoding = network.Transformer(
            attention_layer=3,
            attention_head=2,
            embed_dim=self.embed_dim,
            use_token=True,
        )(embedding)

        # map encoding to corresponding place
        map = network.encoding_to_map(encoding, tower_coord, self.map_size)

        # encode map
        map = network.Preprocessor(post_channel=16, is_2d=True)(map)
        map = network.SpacialEncoder([32, 64, 64])(map)

        map = jnp.reshape(map, [map.shape[0], -1])
        map = network.DenseStack(
            [9 * self.embed_dim, 6 * self.embed_dim, 3 * self.embed_dim]
        )(map)

        # cls - [batch, embed_dim]
        # map - [batch, 3*embed_dim]

        pass
