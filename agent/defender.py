import flax.linen as nn
import jax.numpy as jnp
import jax


from typing import Any, Tuple

from agent import network


class SpacialEncoder(nn.Module):
    """
    Attributes:
        embed_size: size used for embedding tower attributes
        map_size: (height, width) of game map, used for mapping tower to map.
    """

    embed_size: int
    map_size: Tuple[int, int] = (256, 256)

    @nn.compact
    def __call__(self, tower):
        """
        Args:
            tower: tuple of (tower_attr, tower_coord), with the dimension of ([batch, length, attr], [batch, length, 2])

        Returns:
            tower_cls: [batch, embed_size]
            map: [batch, map_height/8, map_width/8, 64]
            map_encoding: [batch, embed_size]
        """

        tower_attr, tower_coord = tower

        # embed tower attribute using Embedder
        tower_embedding = network.Embedder(embed_size=self.embed_size)(tower_attr)

        # apply Transformer to embedding
        tower_cls, tower_encoding = network.Transformer(
            attention_layer=3,
            attention_head=2,
            embed_size=self.embed_size,
            use_token=True,
        )(tower_embedding)

        # map encoding to corresponding place
        map = network.encoding_to_map(tower_encoding, tower_coord, self.map_size)

        # encode map
        map = network.Preprocessor(post_channel=16, is_2d=True)(map)
        map = network.SpacialEncoder([32, 64, 64])(map)

        # here map is later returned for PositionSelector as skip connection
        # map is further processed with MLP to get map_encoding
        map_encoding = jnp.reshape(map, [map.shape[0], -1])
        map_encoding = network.DenseStack(
            [
                4 * self.embed_size,
                2 * self.embed_size,
                self.embed_size,
            ]
        )(map_encoding)

        return tower_cls, map, map_encoding


class ScalarEncoder(nn.Module):
    """remaining money, entity distribution ect."""

    embed_size: int

    @nn.compact
    def __call__(self, scalar):
        """
        Args:
            scalar: [batch, scalar_attr]

        Returns:
            [batch, embed_size]
        """
        embedding = network.Embedder(self.embed_size)(scalar)
        return embedding


class TowerSelector(nn.Module):
    hidden_size: int
    tower_types: int

    temperature: float

    @nn.compact
    def __call__(self, inputs, hidden_state, key):
        """
        Args:
            inputs: [batch, 3*embed_size]
            key: PRNGKey
            hidden_state: (cell, hidden) where dimension is ([batch, hidden_size], [batch, hidden_size])

        Returns:
            next_hidden_state: next hidden state, dimension same as Args
            tower_logit: logit of tower, [batch, tower_types]
            tower_sample: index of sampled towers, [batch]
            autoregressive_encoding: [batch, hidden_size]
        """
        batch = inputs.shape[0]

        # use sos token if hidden_state is not provided
        if hidden_state == None:
            sos_cell = self.param(
                "sos_cell", nn.initializers.zeros, (1, self.hidden_size)
            )
            sos_cell = jnp.tile(sos_cell, [batch, 1])
            sos_hidden = self.param(
                "sos_hidden", nn.initializers.zeros, (1, self.hidden_size)
            )
            sos_hidden = jnp.tile(sos_hidden, [batch, 1])
            hidden_state = (sos_cell, sos_hidden)

        # LSTM
        next_hidden_state, lstm_out = nn.LSTMCell()(hidden_state, inputs)

        # calculate prob
        key, subkey = jax.random.split(key)
        tower_logits = nn.Dense(features=self.tower_types)(lstm_out)
        tower_logits = tower_logits / self.temperature

        # sample and create one-hot encoding of sampled tower
        batch_idx = jnp.arange(batch)
        tower_sample = jax.random.categorical(subkey, tower_logits)
        one_hot = jnp.zeros((batch, self.tower_types))  # [batch, tower_types]
        one_hot[batch_idx, tower_sample] = 1

        # calculate autoregressive_encoding
        autoregressive_encoding = nn.Dense(self.hidden_size)(one_hot)
        autoregressive_encoding = autoregressive_encoding + lstm_out
        autoregressive_encoding = nn.relu(autoregressive_encoding)
        autoregressive_encoding = nn.Dense(self.hidden_size)(autoregressive_encoding)
        autoregressive_encoding = nn.relu(autoregressive_encoding)

        return next_hidden_state, tower_logits, tower_sample, autoregressive_encoding


class PositionSelector(nn.Module):
    temperature: float

    @nn.compact
    def __call__(self, map, autoregressive_encoding, key):
        """
        Args:
            map:
            autoregressive_encoding:

        Returns:
            position_logits: [batch, height, width]
            position_sample: [batch, 2]
        """

        batch, hidden_size = autoregressive_encoding

        # reshape autoregressive_encoding to match the shape of map
        encoding = jnp.reshape(
            autoregressive_encoding, [batch, hidden_size / 2, hidden_size / 2, 1]
        )
        kernel_size = (
            map.shape[1] - encoding.shape[1] + 1,
            map.shape[2] - encoding.shape[2] + 1,
        )
        encoding = nn.ConvTranspose(
            features=4, kernel_size=kernel_size, strides=(1, 1), padding="VALID"
        )(encoding)
        encoding = nn.relu(encoding)

        # concatenate encoding and map
        map = jnp.concatenate([encoding, map], axis=-1)

        # rescale back to original map size
        map = nn.ConvTranspose(
            features=32, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
        )(map)
        map = nn.relu(map)
        map = nn.ConvTranspose(
            features=16, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
        )(map)
        map = nn.relu(map)
        map = nn.ConvTranspose(
            features=1, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
        )(map)

        # get logit and sample
        key, subkey = jax.random.split(key)
        batch, height, width, _ = map.shape

        position_logits = jnp.reshape(map, [batch, height, width])
        position_logits = position_logits / self.temperature

        logit_flatten = jnp.reshape(position_logits, [batch, -1])
        position_idx = jax.random.categorical(subkey, logit_flatten)
        (position_x, position_y) = (position_idx / width, position_idx % width)
        position_sample = jnp.concatenate([position_x, position_y], axis=-1)

        return position_logits, position_sample


class DefenderBuilder(nn.Module):
    embed_size: int
    hidden_size: int
    temperature: float
    tower_types: int
    map_size: Tuple[int, int] = (256, 256)

    @nn.compact
    def __call__(self, key, tower, scalar, hidden_state=None):
        """

        Args:
            key: PRNGKey
            tower: tuple of (tower_attr, tower_coord), with the dimension of ([batch, length, attr], [batch, length, 2])
            scalar: [batch, scalar_attr]
            hidden_state: (cell, hidden) where dimension is ([batch, embed_size], [batch, embed_size])

        Returns:
            next_hidden_state:
            tower_logits:
            tower_sample:
            position_logits:
            position_sample:
        """

        tower_cls, map, map_encoding = SpacialEncoder(
            embed_size=self.embed_size, map_size=self.map_size
        )(tower)

        scalar_encoding = ScalarEncoder(self.embed_size)(scalar)

        building_selector_input = jnp.concatenate(
            [tower_cls, map_encoding, scalar_encoding], axis=1
        )

        key, subkey = jax.random.split(key)
        (
            next_hidden_state,
            tower_logits,
            tower_sample,
            autoregressive_encoding,
        ) = TowerSelector(
            hidden_size=self.hidden_size,
            tower_types=self.tower_types,
            temperature=self.temperature,
        )(
            inputs=building_selector_input, hidden_state=hidden_state, key=subkey
        )

        position_logits, position_sample = PositionSelector(
            temperature=self.temperature
        )(map=map, autoregressive_encoding=autoregressive_encoding)

        return (
            next_hidden_state,
            tower_logits,
            tower_sample,
            position_logits,
            position_sample,
        )
