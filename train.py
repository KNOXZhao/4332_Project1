import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import sqrt
from torch.utils.data import Dataset, DataLoader

# For tokenization, we reuse Keras' Tokenizer for simplicity.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------
# 1. Load Data
# -----------------------
train_df = pd.read_csv("review.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("prediction.csv")

with open("product.json", "r") as f:
    product_data = json.load(f)
product_df = pd.DataFrame(product_data)

# -----------------------
# 2. Merge Product Metadata
# -----------------------
def extract_main_category(cat_list):
    if not cat_list:
        return "Unknown"
    # Use the last category that is not an HTML artifact.
    last_cat = cat_list[-1]
    if isinstance(last_cat, str) and ('<' in last_cat or '>' in last_cat):
        if len(cat_list) >= 2:
            last_cat = cat_list[-2]
    return last_cat if last_cat else "Unknown"

product_df["MainCategory"] = product_df["category"].apply(extract_main_category)
product_df["BrandClean"] = product_df["brand"].fillna("Unknown")

for df in [train_df, val_df, test_df]:
    df = df.merge(product_df[["ProductID", "MainCategory", "BrandClean"]], on="ProductID", how="left")
    df[["MainCategory", "BrandClean"]] = df[["MainCategory", "BrandClean"]].fillna("Unknown")

train_df = train_df.merge(product_df[["ProductID", "MainCategory", "BrandClean"]], on="ProductID", how="left")
val_df = val_df.merge(product_df[["ProductID", "MainCategory", "BrandClean"]], on="ProductID", how="left")
test_df = test_df.merge(product_df[["ProductID", "MainCategory", "BrandClean"]], on="ProductID", how="left")

for df in [train_df, val_df, test_df]:
    df["MainCategory"].fillna("Unknown", inplace=True)
    df["BrandClean"].fillna("Unknown", inplace=True)

# -----------------------
# 3. Text Tokenization
# -----------------------
# Use the "Text" column from training data. (If you wish, you can also combine "Summary".)
train_texts = train_df["Text"].fillna("").astype(str).tolist()
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
MAX_SEQ_LEN = 100
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# For validation and test, assume no review text is provided (or use empty strings)
val_padded = np.zeros((len(val_df), MAX_SEQ_LEN), dtype=int)
test_padded = np.zeros((len(test_df), MAX_SEQ_LEN), dtype=int)

# -----------------------
# 4. Encode IDs and Categorical Features
# -----------------------
# User and Product IDs
unique_users = train_df["ReviewerID"].unique().tolist()
unique_items = train_df["ProductID"].unique().tolist()
user_to_index = {u: i+1 for i, u in enumerate(unique_users)}  # reserve 0 for unknown
item_to_index = {p: i+1 for i, p in enumerate(unique_items)}
unknown_user_idx = 0
unknown_item_idx = 0

train_user_idx = train_df["ReviewerID"].apply(lambda x: user_to_index.get(x, unknown_user_idx)).values
train_item_idx = train_df["ProductID"].apply(lambda x: item_to_index.get(x, unknown_item_idx)).values
val_user_idx = val_df["ReviewerID"].apply(lambda x: user_to_index.get(x, unknown_user_idx)).values
val_item_idx = val_df["ProductID"].apply(lambda x: item_to_index.get(x, unknown_item_idx)).values
test_user_idx = test_df["ReviewerID"].apply(lambda x: user_to_index.get(x, unknown_user_idx)).values
test_item_idx = test_df["ProductID"].apply(lambda x: item_to_index.get(x, unknown_item_idx)).values

# Categories and Brands
unique_cats = pd.unique(train_df["MainCategory"]).tolist()
unique_brands = pd.unique(train_df["BrandClean"]).tolist()
cat_to_index = {c: i+1 for i, c in enumerate(unique_cats)}  # 0 for unknown
brand_to_index = {b: i+1 for i, b in enumerate(unique_brands)}
unknown_cat_idx = 0
unknown_brand_idx = 0

train_cat_idx = train_df["MainCategory"].apply(lambda x: cat_to_index.get(x, unknown_cat_idx)).values
train_brand_idx = train_df["BrandClean"].apply(lambda x: brand_to_index.get(x, unknown_brand_idx)).values
val_cat_idx = val_df["MainCategory"].apply(lambda x: cat_to_index.get(x, unknown_cat_idx)).values
val_brand_idx = val_df["BrandClean"].apply(lambda x: brand_to_index.get(x, unknown_brand_idx)).values
test_cat_idx = test_df["MainCategory"].apply(lambda x: cat_to_index.get(x, unknown_cat_idx)).values
test_brand_idx = test_df["BrandClean"].apply(lambda x: brand_to_index.get(x, unknown_brand_idx)).values

# -----------------------
# 5. Prepare Target Ratings
# -----------------------
train_ratings = train_df["Star"].values.astype(np.float32)
val_ratings = val_df["Star"].values.astype(np.float32)

# -----------------------
# 6. Create a PyTorch Dataset
# -----------------------
class RatingDataset(Dataset):
    def __init__(self, user_ids, item_ids, text_seqs, cat_ids, brand_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.text_seqs = torch.tensor(text_seqs, dtype=torch.long)
        self.cat_ids = torch.tensor(cat_ids, dtype=torch.long)
        self.brand_ids = torch.tensor(brand_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return (self.user_ids[idx],
                self.item_ids[idx],
                self.text_seqs[idx],
                self.cat_ids[idx],
                self.brand_ids[idx],
                self.ratings[idx])
                
train_dataset = RatingDataset(train_user_idx, train_item_idx, train_padded, train_cat_idx, train_brand_idx, train_ratings)
val_dataset = RatingDataset(val_user_idx, val_item_idx, val_padded, val_cat_idx, val_brand_idx, val_ratings)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------
# 7. Define the Model in PyTorch
# -----------------------
class RatingPredictor(nn.Module):
    def __init__(self, num_users, num_items, vocab_size, num_cats, num_brands, max_seq_len,
                 user_emb_dim=32, item_emb_dim=32, word_emb_dim=100, cat_emb_dim=8, brand_emb_dim=8):
        super(RatingPredictor, self).__init__()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.item_emb = nn.Embedding(num_items, item_emb_dim)
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim)
        self.cat_emb = nn.Embedding(num_cats, cat_emb_dim)
        self.brand_emb = nn.Embedding(num_brands, brand_emb_dim)
        # CNN for text
        self.conv1d = nn.Conv1d(in_channels=word_emb_dim, out_channels=50, kernel_size=3)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Calculate combined feature dimension:
        # user + item + text (50) + category + brand
        combined_dim = user_emb_dim + item_emb_dim + 50 + cat_emb_dim + brand_emb_dim
        self.fc1 = nn.Linear(combined_dim, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(32, 1)
        
    def forward(self, user_ids, item_ids, text_seq, cat_ids, brand_ids):
        user_vec = self.user_emb(user_ids)       # (batch, 32)
        item_vec = self.item_emb(item_ids)       # (batch, 32)
        # Process text:
        # text_seq: (batch, max_seq_len) -> word embeddings: (batch, max_seq_len, word_emb_dim)
        text_emb = self.word_emb(text_seq)       # (batch, max_seq_len, word_emb_dim)
        # Rearrange to (batch, word_emb_dim, max_seq_len) for Conv1d
        text_emb = text_emb.transpose(1, 2)
        conv_out = F.relu(self.conv1d(text_emb))  # (batch, 50, L) where L = max_seq_len - 3 + 1
        pooled = self.maxpool(conv_out).squeeze(-1)  # (batch, 50)
        
        cat_vec = self.cat_emb(cat_ids)          # (batch, cat_emb_dim)
        brand_vec = self.brand_emb(brand_ids)      # (batch, brand_emb_dim)
        
        # Concatenate all features
        combined = torch.cat([user_vec, item_vec, pooled, cat_vec, brand_vec], dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        out = self.fc_out(x)
        return out.squeeze(1)  # (batch,)
    
# Define sizes (add +1 to account for unknown index 0)
num_users = len(user_to_index) + 1
num_items = len(item_to_index) + 1
vocab_size = min(10000, len(tokenizer.word_index) + 1)
num_cats = len(cat_to_index) + 1
num_brands = len(brand_to_index) + 1

model = RatingPredictor(num_users, num_items, vocab_size, num_cats, num_brands, MAX_SEQ_LEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------
# 8. Training Setup
# -----------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 5

def evaluate(model, loader):
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            user_ids, item_ids, text_seqs, cat_ids, brand_ids, ratings = batch
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            text_seqs = text_seqs.to(device)
            cat_ids = cat_ids.to(device)
            brand_ids = brand_ids.to(device)
            ratings = ratings.to(device)
            outputs = model(user_ids, item_ids, text_seqs, cat_ids, brand_ids)
            loss = criterion(outputs, ratings)
            losses.append(loss.item() * ratings.size(0))
            preds.extend(outputs.cpu().numpy())
            trues.extend(ratings.cpu().numpy())
    avg_loss = np.sum(losses) / len(loader.dataset)
    rmse_val = sqrt(np.mean((np.array(preds) - np.array(trues)) ** 2))
    return avg_loss, rmse_val

# -----------------------
# 9. Training Loop
# -----------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        user_ids, item_ids, text_seqs, cat_ids, brand_ids, ratings = batch
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        text_seqs = text_seqs.to(device)
        cat_ids = cat_ids.to(device)
        brand_ids = brand_ids.to(device)
        ratings = ratings.to(device)
        
        optimizer.zero_grad()
        outputs = model(user_ids, item_ids, text_seqs, cat_ids, brand_ids)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * ratings.size(0)
        
    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_rmse = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} -- Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val RMSE: {val_rmse:.4f}")

# -----------------------
# 10. Generate Predictions
# -----------------------
def generate_predictions(model, user_ids, item_ids, text_seqs, cat_ids, brand_ids):
    model.eval()
    inputs = {
        "user_ids": torch.tensor(user_ids, dtype=torch.long).to(device),
        "item_ids": torch.tensor(item_ids, dtype=torch.long).to(device),
        "text_seqs": torch.tensor(text_seqs, dtype=torch.long).to(device),
        "cat_ids": torch.tensor(cat_ids, dtype=torch.long).to(device),
        "brand_ids": torch.tensor(brand_ids, dtype=torch.long).to(device)
    }
    with torch.no_grad():
        preds = model(inputs["user_ids"], inputs["item_ids"], inputs["text_seqs"],
                      inputs["cat_ids"], inputs["brand_ids"])
    return preds.cpu().numpy()

# Predictions for validation set
val_preds = generate_predictions(model, val_user_idx, val_item_idx, val_padded, val_cat_idx, val_brand_idx)
# Compute RMSE manually (or use evaluate.py later)
val_rmse = sqrt(np.mean((val_preds - val_ratings) ** 2))
print(f"Final Validation RMSE: {val_rmse:.4f}")

# Save validation predictions to CSV (for evaluate.py)
val_pred_df = pd.DataFrame({
    "ReviewerID": val_df["ReviewerID"],
    "ProductID": val_df["ProductID"],
    "Star": val_preds
})
val_pred_df.to_csv("validation_prediction.csv", index=False)

# Predictions for test set
test_preds = generate_predictions(model, test_user_idx, test_item_idx, test_padded, test_cat_idx, test_brand_idx)
test_pred_df = pd.DataFrame({
    "ReviewerID": test_df["ReviewerID"],
    "ProductID": test_df["ProductID"],
    "Star": test_preds
})
test_pred_df.to_csv("prediction_filled.csv", index=False)
print("Saved validation_prediction.csv and prediction_filled.csv")
