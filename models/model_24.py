from models.primitives import new_conv2d
from models.primitives import new_res_block_collection_v2

import tensorflow as tf

class TransformerCNNBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
def model_24(WIDTH, HEIGHT):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3), dtype=tf.float16)
    do0 = new_conv2d(input, 96, (7, 7), (5, 5))
    do0 = new_conv2d(do0,256,(5,5),(3,3))
    do0 = new_conv2d(do0,384,(3,3),(2,2))
    do0 = new_conv2d(do0,384,(3,3),(2,2))
    do0 = new_conv2d(do0,384,(3,3),(2,2))

    flat = tf.keras.layers.Flatten()(do0)

    # embed_dim = 32  # Embedding size for each token
    # num_heads = 2  # Number of attention heads
    # ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    # embedding_layer = TokenAndPositionEmbedding(3072, 3072, embed_dim)
    # x = embedding_layer(flat)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(20, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    # embedding_layer = TokenAndPositionEmbedding(3072, 3072, embed_dim)
    # x = embedding_layer(x)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(20, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    
    # embedding_layer = TokenAndPositionEmbedding(3072, 3072, embed_dim)
    # x = embedding_layer(x)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(20, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    # embedding_layer = TokenAndPositionEmbedding(3072, 3072, embed_dim)
    # x = embedding_layer(x)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(20, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    # embedding_layer = TokenAndPositionEmbedding(3072, 3072, embed_dim)
    # x = embedding_layer(x)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(20, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    # embedding_layer = TokenAndPositionEmbedding(3072, 3072, embed_dim)
    # x = embedding_layer(x)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(20, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    # embedding_layer = TokenAndPositionEmbedding(3072, 3072, embed_dim)
    # x = embedding_layer(x)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(20, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    # embedding_layer = TokenAndPositionEmbedding(3072, 3072, embed_dim)
    # x = embedding_layer(x)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(20, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Dense(256, kernel_initializer="he_uniform", activation="relu", dtype=tf.float16)(flat)
    output = tf.keras.layers.Dense(128, kernel_initializer="he_uniform", activation="relu", dtype=tf.float16)(output)
    output = tf.keras.layers.Dense(64, kernel_initializer="he_uniform", activation="relu", dtype=tf.float16)(output)
    output = tf.keras.layers.Dense(1, kernel_initializer="he_uniform", activation="sigmoid", dtype=tf.float16)(output)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model