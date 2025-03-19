import pandas as pd
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load Data
train_df = pd.read_csv("review.csv")        # training reviews dataset
val_df = pd.read_csv("validation.csv")      # validation set with actual ratings
test_df = pd.read_csv("prediction.csv")     # test set with missing ratings

# Load product metadata
with open("product.json", "r") as f:
    product_data = json.load(f)
product_df = pd.DataFrame(product_data)

# 2. Merge metadata into training reviews
# Extract a single category and brand for each product for simplicity
# We'll use the last category in the list (if exists) and the brand string.
def extract_main_category(cat_list):
    if not cat_list: 
        return "Unknown"
    # Take the last category that is not an HTML artifact
    last_cat = cat_list[-1]
    if isinstance(last_cat, str) and ('<' in last_cat or '>' in last_cat):
        # If the last entry is an HTML tag like "</span>", take the second last
        if len(cat_list) >= 2:
            last_cat = cat_list[-2]
    return last_cat if last_cat else "Unknown"

product_df["MainCategory"] = product_df["category"].apply(extract_main_category)
product_df["BrandClean"] = product_df["brand"].fillna("Unknown")
# Merge into train, validation, and test data
train_df = train_df.merge(product_df[["ProductID","MainCategory","BrandClean"]], on="ProductID", how="left")
val_df = val_df.merge(product_df[["ProductID","MainCategory","BrandClean"]], on="ProductID", how="left")
test_df = test_df.merge(product_df[["ProductID","MainCategory","BrandClean"]], on="ProductID", how="left")

# Fill any missing merges with "Unknown"
train_df[["MainCategory","BrandClean"]] = train_df[["MainCategory","BrandClean"]].fillna("Unknown")
val_df[["MainCategory","BrandClean"]] = val_df[["MainCategory","BrandClean"]].fillna("Unknown")
test_df[["MainCategory","BrandClean"]] = test_df[["MainCategory","BrandClean"]].fillna("Unknown")

# 3. Prepare text data
# Combine 'Text' and 'Summary' (optional) or use 'Text' alone for training reviews
train_texts = train_df["Text"].fillna("").astype(str).tolist()
# Fit a tokenizer on the training review text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")  # limit vocab size for efficiency
tokenizer.fit_on_texts(train_texts)

# Convert texts to sequences and pad them
MAX_SEQ_LEN = 100  # maximum words to consider from each review
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# For validation and test, if we had review text we would tokenize similarly.
# In our dataset, validation/test files do not include the review text, only IDs and Star.
# We will assume we don't have text for those (if we did, we would apply tokenizer and pad as above).

# Instead, we'll use an empty text or placeholder for those to feed into the model.
val_padded = np.zeros((len(val_df), MAX_SEQ_LEN), dtype=int)
test_padded = np.zeros((len(test_df), MAX_SEQ_LEN), dtype=int)
# (Alternatively, we could skip using text features for validation/test if no text is provided.
# The model will rely on user/item embeddings and metadata in those cases.)

# 4. Encode user IDs and product IDs
# Build mappings from ID strings to numeric indices
unique_users = train_df["ReviewerID"].unique().tolist()
unique_items = train_df["ProductID"].unique().tolist()
# Include unknown tokens for any new user/item
user_to_index = {u: i for i, u in enumerate(unique_users, start=1)}  # reserve 0 for unknown
item_to_index = {p: i for i, p in enumerate(unique_items, start=1)}
unknown_user_idx = 0
unknown_item_idx = 0

# Prepare user and item id arrays for train
train_user_idx = train_df["ReviewerID"].apply(lambda x: user_to_index.get(x, unknown_user_idx)).values
train_item_idx = train_df["ProductID"].apply(lambda x: item_to_index.get(x, unknown_item_idx)).values
# For validation and test, map with the same dictionary, use 0 if not found (unknown)
val_user_idx = val_df["ReviewerID"].apply(lambda x: user_to_index.get(x, unknown_user_idx)).values
val_item_idx = val_df["ProductID"].apply(lambda x: item_to_index.get(x, unknown_item_idx)).values
test_user_idx = test_df["ReviewerID"].apply(lambda x: user_to_index.get(x, unknown_user_idx)).values
test_item_idx = test_df["ProductID"].apply(lambda x: item_to_index.get(x, unknown_item_idx)).values

# 5. Encode category and brand
unique_cats = pd.unique(train_df["MainCategory"]).tolist()
unique_brands = pd.unique(train_df["BrandClean"]).tolist()
cat_to_index = {c: i for i, c in enumerate(unique_cats, start=1)}  # 0 for unknown
brand_to_index = {b: i for i, b in enumerate(unique_brands, start=1)}
unknown_cat_idx = 0
unknown_brand_idx = 0

train_cat_idx = train_df["MainCategory"].apply(lambda x: cat_to_index.get(x, unknown_cat_idx)).values
train_brand_idx = train_df["BrandClean"].apply(lambda x: brand_to_index.get(x, unknown_brand_idx)).values
val_cat_idx = val_df["MainCategory"].apply(lambda x: cat_to_index.get(x, unknown_cat_idx)).values
val_brand_idx = val_df["BrandClean"].apply(lambda x: brand_to_index.get(x, unknown_brand_idx)).values
test_cat_idx = test_df["MainCategory"].apply(lambda x: cat_to_index.get(x, unknown_cat_idx)).values
test_brand_idx = test_df["BrandClean"].apply(lambda x: brand_to_index.get(x, unknown_brand_idx)).values

# 6. Prepare target arrays
train_ratings = train_df["Star"].values.astype('float32')
val_ratings   = val_df["Star"].values.astype('float32')

# 7. Define the model architecture
# Input layers
user_input = Input(shape=(), name="user_id")
item_input = Input(shape=(), name="item_id")
text_input = Input(shape=(MAX_SEQ_LEN,), name="review_text")
cat_input  = Input(shape=(), name="category")
brand_input= Input(shape=(), name="brand")

# Embedding layers
num_users = len(user_to_index) + 1  # +1 because we started indices at 1 and include 0 as unknown
num_items = len(item_to_index) + 1
user_emb = Embedding(input_dim=num_users, output_dim=32, embeddings_initializer='he_normal', name="user_emb")
item_emb = Embedding(input_dim=num_items, output_dim=32, embeddings_initializer='he_normal', name="item_emb")

user_vec = user_emb(user_input)    # shape: (None, 32)
item_vec = item_emb(item_input)    # shape: (None, 32)
user_vec = keras.layers.Flatten()(user_vec)  # Flatten to (None, 32)
item_vec = keras.layers.Flatten()(item_vec)

# Text embedding + CNN
vocab_size = min(10000, len(tokenizer.word_index)+1)
text_emb_layer = Embedding(input_dim=vocab_size, output_dim=100, embeddings_initializer='uniform', name="word_emb")
text_embedded = text_emb_layer(text_input)  # (None, MAX_SEQ_LEN, 100)
# Convolutional layer to extract features
conv = Conv1D(filters=50, kernel_size=3, activation='relu')(text_embedded)
text_conv_out = GlobalMaxPooling1D()(conv)  # (None, 50) vector from text

# Category and Brand embeddings
num_cats = len(cat_to_index) + 1
num_brands = len(brand_to_index) + 1
cat_emb_layer = Embedding(input_dim=num_cats, output_dim=8, embeddings_initializer='he_normal', name="cat_emb")
brand_emb_layer = Embedding(input_dim=num_brands, output_dim=8, embeddings_initializer='he_normal', name="brand_emb")
cat_vec = keras.layers.Flatten()(cat_emb_layer(cat_input))       # (None, 8)
brand_vec = keras.layers.Flatten()(brand_emb_layer(brand_input)) # (None, 8)

# Concatenate all feature vectors
combined = Concatenate()([user_vec, item_vec, text_conv_out, cat_vec, brand_vec])
# Add dense layers for feature interaction
x = Dense(64, activation='relu')(combined)
x = Dropout(0.5)(x)  # dropout to reduce overfitting
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
# Output layer for rating
output = Dense(1, activation='linear')(x)

model = keras.Model(inputs=[user_input, item_input, text_input, cat_input, brand_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])

# 8. Train the model with validation
model.fit(
    x = {
        "user_id": train_user_idx,
        "item_id": train_item_idx,
        "review_text": train_padded,
        "category": train_cat_idx,
        "brand": train_brand_idx
    },
    y = train_ratings,
    batch_size = 64,
    epochs = 5,
    validation_data = (
        {
          "user_id": val_user_idx,
          "item_id": val_item_idx,
          "review_text": val_padded,
          "category": val_cat_idx,
          "brand": val_brand_idx
        },
        val_ratings
    ),
    verbose = 2
)
# Note: In practice, increase epochs and use EarlyStopping callback for better tuning.

# 9. Evaluate on validation set
val_predictions = model.predict({
    "user_id": val_user_idx,
    "item_id": val_item_idx,
    "review_text": val_padded,
    "category": val_cat_idx,
    "brand": val_brand_idx
})
val_predictions = val_predictions.flatten()
rmse = np.sqrt(np.mean((val_predictions - val_ratings) ** 2))
print(f"Validation RMSE: {rmse:.4f}")

# 10. Generate predictions for test set
test_predictions = model.predict({
    "user_id": test_user_idx,
    "item_id": test_item_idx,
    "review_text": test_padded,
    "category": test_cat_idx,
    "brand": test_brand_idx
})
test_predictions = test_predictions.flatten()

# 11. Save predictions to CSV
# Create DataFrames for output
val_pred_df = pd.DataFrame({
    "ReviewerID": val_df["ReviewerID"],
    "ProductID": val_df["ProductID"],
    "Star": val_predictions
})
test_pred_df = pd.DataFrame({
    "ReviewerID": test_df["ReviewerID"],
    "ProductID": test_df["ProductID"],
    "Star": test_predictions
})
# Ensure the columns are in the correct order and format
val_pred_df.to_csv("validation_prediction.csv", index=False)
test_pred_df.to_csv("prediction_filled.csv", index=False)
print("Saved validation_prediction.csv and prediction_filled.csv")
