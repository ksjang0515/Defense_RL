from typing import Any
import flax.linen as nn

import jax.numpy as jnp


class Embedder(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, inputs):
        """Embeds using one Dense layer and ReLU.

        Args:
            inputs: [batch, length, attr]

        Returns:
            tensor of [batch, length, embed_dim]
        """
        x = inputs
        dtype = x.dtype

        x = nn.Dense(features=self.embed_dim, dtype=dtype)
        x = nn.relu(x)

        return x


class Transformer(nn.Module):
    attention_layer: int
    attention_head: int

    embed_dim: int
    use_token: bool = True

    @nn.compact
    def __call__(self, inputs, deterministic):
        """Applies Self-Attention mechanism.

        Args:
            inputs: [batch, length, embed_dim]

        Returns:
            if use_token
                cls token of size [batch, embed_dim]
                encoding of size [batch, length, embed_dim]

            else
                encoding of size [batch, length, embed_dim]

        """
        x = inputs
        dtype = x.dtype

        if self.use_token:
            b, _, _ = x.shape

            cls = self.param("cls", nn.initializers.zeros, (1, 1, self.embed_dim))
            cls = jnp.tile(cls, [b, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)

        # apply transformer to embedded tower
        for _ in range(self.attention_layer):
            x = nn.SelfAttention(
                num_heads=self.attention_head,
                dtype=dtype,
                deterministic=deterministic,
            )(x)

        if self.use_token:
            cls = x[:, 0]
            encoding = x[:, 1:]

            return cls, encoding

        return x


def encoding_to_map(encoding, coordinate, map_size):
    """Maps encoding to corresponding index on the map. Empty spaces are represented as zeros.

    Args:
        inputs: [batch, length, embed_dim]
        coordinate: [batch, length, 2], where last dimension is (x, y) or (height, width)
        map_size: (height, width)

    Returns:
        tensor of [batch, map_height, map_width, embed_dim]

    """

    map_width, map_height = map_size
    batch_size, _, embed_dim = encoding.shape
    dtype = encoding.dtype

    # place encoding to corresponding coordinate
    spacial_map = jnp.zeros((batch_size, map_height, map_width, embed_dim), dtype=dtype)

    coordinate_x = coordinate[:, :, 0]
    coordinate_y = coordinate[:, :, 1]

    spacial_map[:, coordinate_x, coordinate_y] = encoding

    return spacial_map


class SpacialEncoder(nn.Module):
    channel: list[int]

    @nn.compact
    def __call__(self, inputs):
        """Applies layers of 2d CNN to reduce the size by power of 2.

        Args:
            inputs: [batch, height, width, channel]

        Returns:
            tensor of [batch, height/n, width/n, last_channel], where n corresponds to 2^len(channel)
        """
        x = inputs
        dtype = x.dtype

        for ch in self.channel:
            x = nn.Conv(
                features=ch,
                kernel_size=(4, 4),
                padding="SAME",
                strides=2,
                dtype=dtype,
            )(x)
            x = nn.relu(x)

        return x


class DenseStack(nn.Module):
    features: list[int]

    @nn.compact
    def __call__(self, inputs):
        """Applies layers of MLP and ReLU.

        Args:
            inputs: [batch, *, feature]

        Returns:
            tensor of [batch, *, last_feature]
        """
        x = inputs
        dtype = x.dtype

        for f in self.features:
            x = nn.Dense(
                features=f,
                dtype=dtype,
            )(x)
            x = nn.relu(x)

        return x


class Preprocessor(nn.Module):
    post_channel: int
    is_2d: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Applies preprocessing to inputs by using CNN with kernel size 1

        Args:
            inputs: [batch, *, channel]

        Returns:
            tensor of [batch, *, post_channel]
        """

        x = inputs
        dtype = x.dtype
        kernel_size = (1, 1) if self.is_2d else (1,)

        x = nn.Conv(
            features=self.post_channel,
            kernel_size=kernel_size,
            padding="VALID",
            dtype=dtype,
        )
        x = nn.relu(x)

        return x
