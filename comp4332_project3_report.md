# COMP4332 Project 1 Report

**Group 5:** YUNG Ka Shing, ZHAO Yubo, Tam Kiu Wai, Nguyen Kim Hue Nam 

## 1. Introduction

For this project, we used two approaches: **Neural CF** and **Wide & Deep Learning**. 

This report details our methodology, implementation choices, experimental results, and key insights gained throughout the process. Our best-performing model achieved **RMSE** of **0.8441** on the validation set.

<!-- [Photo: Dataset Overview]
*Include a visualization showing the distribution of ratings in the dataset (histogram) and possibly a heatmap of user-item interactions to illustrate the sparsity of the data.* -->

## 2. Neural CF Model

Our best-performing approach used **Neural CF**.

### 2.1 Data Analysis and Cold Start Problem

We began by analyzing the validation and test sets to understand the prevalence of new users and items:

```python
# Validation set analysis:
Number of rows in validation set with both ProductID and ReviewerID not in training set: 42
Number of rows in validation set with only ProductID not in training set: 1101
Number of rows in validation set with only ReviewerID not in training set: 359
length of new users: 47
length of new items: 302
ratio: 6.425531914893617

# Test set analysis:
Number of rows in test set with both ProductID and ReviewerID not in training set: 34
Number of rows in test set with only ProductID not in training set: 1166
Number of rows in test set with only ReviewerID not in training set: 349
length of new users: 42
length of new items: 311
ratio: 7.404761904761905
```
This analysis revealed an imbalance in the cold-start problem, which informed our masking strategy.

Our implementation's core innovation was the effective initialization and learning of **embeddings** to capture user preferences and item characteristics, with a specialized approach for handling unseen entities.

### 2.2 Data Preprocessing

#### 2.2.1 User and Item ID Encoding

The first preprocessing step was converting categorical user and item IDs into numerical indices for model consumption. We reserved index 0 for unknown entities to handle the cold start problem:

```python
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
```

#### 2.2.2 Embedding Approaches

We explored two distinct embedding approaches for representing users and items:

1. **Lookup Table Approach**: Using standard `torch.nn.Embeddings` for both reviewers and products
2. **Lookup Table with Text Encoder**: Employing a text encoder for both user and item embeddings, and a lookup table for reviewer embeddings

The 2nd one outperformed the 1st one, reducing RMSE from 0.9 to **0.8441**. For text encoding, we selected the lightweight **`avsolatorio/GIST-small-Embedding-v0`** model from Hugging Face's MTEB Leaderboard.

#### 2.2.3 Embedding Initialization Details

1. **User Embedding Initialization**
    1. **Group the reviews** according to the reviewer ID
    2. Use a small but accurate enough text embedding model **'avsolatorio/GIST-small-Embedding-v0'** from hugging face to create initial user embeddings of size **384** based of their reivews
    3. Since there will be **unseen users** in validation and test set, their intial embedding will be encoded based on the text **"Unknown_User"**
2. **Item Embedding Initialization**
    1. **Remove features** with empty values in the product information data
    2. **Remove "main_cat"** features as all product shares same values
    3. Use the text embedding model **'avsolatorio/GIST-small-Embedding-v0'** to create initial item embeddings of size **384** based of their product information
    4. Since there will be **unseen items** in validation and test set, their intial embedding will be encoded based on the text **"Unknown_Item"**

### 2.3 Model Architecture (focus on 2nd param sets)

Neural CF has 2 complementary components:

1. **Generalized Matrix Factorization (GMF)**
2. **Multi-Layer Perceptron (MLP)**

The specific architecture includes:
- Shared **embeddings** for users and items (384-dimensional)
- **GMF path**: Element-wise multiplication of user and item embeddings, followed by a linear layer to produce a scalar output
- **MLP path**:
  - Concatenation of user and item embeddings as input (768-dimensional)
  - Multiple fully-connected layers with decreasing dimensions (384 → 192 → output)
  - **Layer normalization** before each fully-connected layer
  - **LeakyReLU** activation (alpha=0.1) for non-linearity
  - **Dropout** rates starting at 0.6 for strong regularization
  - Skip connections to improve gradient flow
- **Fusion layer**: Concatenation of GMF and MLP outputs, followed by a linear layer
- Final activation: Modified **sigmoid** (scaled to output range [1, 5]): `1 + 4 * torch.sigmoid(x)`

[Photo: Neural CF Model Architecture]

Then, concatenate outputs from GMF and MLP as input to dense layer with modified sigmoid function as `1 + 4 * torch.sigmoid(x)`

### 2.4 Training Process

Our training strategy incorporated several advanced techniques:

- **Loss function**: MSELoss
- **Optimizer**: **AdamW**
- **Learning rate**: Different learning rates for different parameters - smaller learning rate of 1e-4 for user and item embeddings (as they are generated from pretrained models), and higher learning rate for other parameters
- **Learning rate scheduler**: Three-stage strategy with `warmup_lr_scheduler` for first 5 epochs, fixed rate until 20 epochs, and finally `CosineAnnealingLR` for remaining epochs
- **Masking Input data**: The input user ID and item ID are masked as 0 (i.e., Unknown user or item) with probability 0.05 and 0.3 respectively to mimic unseen users and items in validation and test sets

<!-- [Photo: Learning Rate Schedule] -->

### 2.5 Evaluation Results

Our Neural CF model achieved an **RMSE** of 0.8441 on the validation set.

Loss and RMSE logs: at epoch 13, the validation rmse is the **lowest**
```python
Epoch 1/50 -- Train Loss: 3.5369  Train RMSE: 1.8807  Val Loss: 3.2495  Val RMSE: 1.8026
Checkpoint saved at epoch 1 with validation rmse 1.8026
Epoch 2/50 -- Train Loss: 1.0728  Train RMSE: 1.0358  Val Loss: 1.0340  Val RMSE: 1.0168
Checkpoint saved at epoch 2 with validation rmse 1.0168
Epoch 3/50 -- Train Loss: 0.9187  Train RMSE: 0.9585  Val Loss: 1.0439  Val RMSE: 1.0217
Epoch 4/50 -- Train Loss: 0.8982  Train RMSE: 0.9477  Val Loss: 0.8851  Val RMSE: 0.9408
Checkpoint saved at epoch 4 with validation rmse 0.9408
Epoch 5/50 -- Train Loss: 0.7169  Train RMSE: 0.8467  Val Loss: 0.7602  Val RMSE: 0.8719
Checkpoint saved at epoch 5 with validation rmse 0.8719
Epoch 6/50 -- Train Loss: 0.6134  Train RMSE: 0.7832  Val Loss: 0.7602  Val RMSE: 0.8719
Epoch 7/50 -- Train Loss: 0.6109  Train RMSE: 0.7816  Val Loss: 0.7602  Val RMSE: 0.8719
Epoch 8/50 -- Train Loss: 0.6148  Train RMSE: 0.7841  Val Loss: 0.7602  Val RMSE: 0.8719
Epoch 9/50 -- Train Loss: 0.6170  Train RMSE: 0.7855  Val Loss: 0.7602  Val RMSE: 0.8719
Epoch 10/50 -- Train Loss: 0.6145  Train RMSE: 0.7839  Val Loss: 0.7602  Val RMSE: 0.8719
Epoch 11/50 -- Train Loss: 0.6077  Train RMSE: 0.7796  Val Loss: 0.7371  Val RMSE: 0.8586
Checkpoint saved at epoch 11 with validation rmse 0.8586
Epoch 12/50 -- Train Loss: 0.5515  Train RMSE: 0.7427  Val Loss: 0.7400  Val RMSE: 0.8603
Epoch 13/50 -- Train Loss: 0.5255  Train RMSE: 0.7249  Val Loss: 0.7125  Val RMSE: 0.8441

Checkpoint saved at epoch 13 with validation rmse 0.8441

Epoch 14/50 -- Train Loss: 0.5052  Train RMSE: 0.7108  Val Loss: 0.7208  Val RMSE: 0.8490
Epoch 15/50 -- Train Loss: 0.4937  Train RMSE: 0.7026  Val Loss: 0.7293  Val RMSE: 0.8540
Epoch 16/50 -- Train Loss: 0.4822  Train RMSE: 0.6944  Val Loss: 0.7344  Val RMSE: 0.8570
...
```

[Photo: Training Curve]

![](images/2025-04-04-23-50-22.png)

## 3. Other Fail Trial

### 3.1 Failed Trial on NeuralCF with Separate Embeddings

In this approach, we modified the Neural CF architecture to use **separate embedding layers** for GMF and MLP components (**192-dimensional** for GMF, **384-dimensional** for MLP) instead of shared embeddings. The model used a **simplified MLP structure** with different normalization order and **no skip connections**. 

:::danger
This approach only achieved an RMSE of **?????** on the validation set.
:::

### 3.2 Failed Trial on Deep model
For users, we employed **"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"** to summarize user preferences, then converted these summaries to **384-dimensional** embeddings using **'all-MiniLM-L6-v2'**. 

For products, we extracted meaningful metadata including **main categories** and **brand information** from `product.json`. Our architecture combined embeddings for users, items, user features, categories, and brands, processing them through **three MLP layers** before applying a modified sigmoid activation `1 + 4 * torch.sigmoid(x)`. This approach yielded a validation RMSE of **0.8858** without input masking, which improved to **0.8756** when we implemented masking strategies.

## 4. Conclusion


We have 3 models:
| Model | Description | Validation RMSE |
|-------|-------------|----------------|
| Neural CF (Shared Embeddings) | best model | **0.8441** |
| Neural CF (Separate Embeddings) | Modified Neural CF with separate embeddings for GMF and MLP, simplified MLP structure | **????** |
| Deep Model with LLM | Deep network using LLM-generated user features and product metadata | 0.8756 |


Our best-performing model achieved an **RMSE** of **0.8441** on the validation set. Thus, we can conclude that the **Neural CF** model with **shared embeddings** is the best model for this dataset.