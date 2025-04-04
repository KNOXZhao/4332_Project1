# Current progress
## NeuralCF (NeuralCF.ipynb):
**Validation RMSE:** 0.8441

**Method**
1. User Embedding Initialization
    1. Group the reviews according to the reviewer ID
    2. Use a small but accurate enough text embedding model 'avsolatorio/GIST-small-Embedding-v0' from hugging face to create initial user embeddings of size 384 based of their reivews
    3. Since there will be unseen users in validation and test set, their intial embedding will be encoded based on the text "Unknown_User"
2. Item Embedding Initialization
    1. Remove features with empty values in the product information data
    2. Remove "main_cat" features as all product shares same values
    3. Use the text embedding model 'avsolatorio/GIST-small-Embedding-v0' to create initial item embeddings of size 384 based of their product information
    4. Since there will be unseen items in validation and test set, their intial embedding will be encoded based on the text "Unknown_Item"
3. Model Architecture (refer to **section "2nd set of param"**)
    1. Neural CF with GMF and MLP
    2. GMF and MLP share the **same** item and user embeddings
    3. concatenate outputs from GMF and MLP as input to dense layer with modified sigmoid function as (1 + 4 * torch.sigmoid(x))
    4. p.s. you may add picture showing the architecture, thxx
4. Training:
    1. Loss function: MSELoss
    2. different learning rate for different parameters, with smaller leanring rate of 1e-4 for user embedding and item embeddings as they are generated from pretrined models
    3. optimizer: AdamW 
    4. learning rate schedular: warmup_lr_scheduler for first 5 epochs, fixed_lr_scheduler until 20 epochs and finally CosineAnnealingLR for remaining epochs
    5. **Masking Input data** the input user id and item id will be masked as 0 (i.e. Unknown user or item) will probability 0.05 and 0.3 to mimic the case that there will be unseen user and item in the validation set and test set
    6. p.s. the ratio of 0.05 to 0.3 is to approximate the ratio of new user and item in the validation set and test set (refer to **section "# Load Data"**)

## Other fail trial on NeuralCF (NeuralCF.ipynb):
1. Model Architecture (refer to **section "3rd set of param"**)
    1. Neural CF with GMF (same) and MLP (with different architecture)
    2. GMF and MLP use **different** item and user embeddings

## Other fail trial on deep model (train.ipynb):
1. Create User feature embedding:
    1. Use LLM model "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" to summarize interests, intentions and product preferences for each user
    2. Use small text embedding model 'all-MiniLM-L6-v2' to generate user feature embeddings of size 384
2. Create product feature:
    1. extract the main category as the last category that is not an HTML artifact and brand from product.json
3. Model Architecture 
    1. Depp metwork with input as embedding for users, items, user features, item categories, item brands
    2. concatenate all input embeddings and then pass to 3 MLP layers
    3. finally pass to dense layer with modified sigmoid function as (1 + 4 * torch.sigmoid(x))
4. Training:
    1. train without masking input: **Validation RMSE** of 0.8858
    2. train with masking input: **Validation RMSE** of 0.8756


## (Fail)Wide and Deep Model (wide_and_deep_Newt.ipynb): 
**Validation RMSE:** 0.878
Remark: only need to have brief introduction on this failed approach, like only including the loss curve graph and final RMSE
