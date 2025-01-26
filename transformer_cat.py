#%% packages

import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#%% categorize y

# def categorize_y(x, y, pred_name="close"):
#     last_column_name = [col for col in x.columns if col.startswith(pred_name)][-1]
#     last_data = x[last_column_name]
#     res = y - last_data
#     res[res >= 0] = 1
#     res[res < 0] = 0
#     return res

#%%  y to increments

def y_to_increments(x, y, pred_name="close"):
    last_column_name = [col for col in x.columns if col.startswith(pred_name)][-1]
    last_data = x[last_column_name]
    res = (y - last_data) / last_data
    return res

def categorize_y(x, y, pred_name="close"):
    y = y_to_increments(x, y, pred_name=pred_name)
    y[y >= 0] = 1
    y[y < 0] = 0
    return y

#%% reading data (y as returns)

def load_dataset(directory_name, name, target_horizon):
    df_file_path = os.path.join(directory_name, f"df_{name}_{target_horizon}.csv")
    y_file_path = os.path.join(directory_name, f"y_{name}_{target_horizon}.csv")
    if os.path.isfile(df_file_path) and os.path.isfile(y_file_path):
        df = pd.read_csv(df_file_path)
        y = pd.read_csv(y_file_path)
        return df, y

directory_name = "datasets"

df_train, y_train = load_dataset(directory_name, "train", 1)
df_val, y_val = load_dataset(directory_name, "val", 1)
df_test, y_test = load_dataset(directory_name, "test", 1)

y_train = categorize_y(df_train, y_train.y)
y_val = categorize_y(df_val, y_val.y)
y_test = categorize_y(df_test, y_test.y)

#%% evaluation

def count_acc(model, x, y):
    y_pred = model(x).detach().numpy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    
    acc = 1 - mean_squared_error(y, y_pred)
    return acc

def evaluate_model_acc(model, x, y, x_set):
    results = {}
    name_columns = [col for col in x_set.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
        name_index = x_set[x_set[f"name_{name}"] == 1].index
        
        y_name = y[name_index]
        x_name = x[name_index]
        
        acc = count_acc(model, x_name, y_name)
        
        results[name] = {"acc": acc}
    return results

#%% plot prediction

def plot_prediction(model, x, y, x_set):
    y_pred = model(x).detach().numpy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    name_columns = [col for col in x_set.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
        name_index = x_set[x_set[f"name_{name}"] == 1].index
        
        y_name = y[name_index]
        pred_name = y_pred[name_index]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(name_index, y_name, alpha=0.7, label='original', color='blue', s=8)
        plt.scatter(name_index, pred_name, label='pred', alpha=0.7, color='orange', s=8)
        plt.title(f'{name} close price prediction', fontsize=16)
        plt.xlabel('index', fontsize=12)
        plt.ylabel('close price', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
        
#%% model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 64

X_train = torch.tensor(df_train.values, dtype=torch.float32)
Y_train = torch.tensor(y_train.values, dtype=torch.float32)
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

X_val = torch.tensor(df_val.values, dtype=torch.float32)
Y_val = torch.tensor(y_val.values, dtype=torch.float32)
val_dataset = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

X_test = torch.tensor(df_test.values, dtype=torch.float32)
Y_test = torch.tensor(y_test.values, dtype=torch.float32)
test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#%%

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(p=0.2)
        self.decoder = nn.Linear(d_model, 1)
        self.batch_norm_1 = nn.BatchNorm1d(d_model)
        self.batch_norm_2 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        x = self.encoder(x)
        x = self.batch_norm_1(x)
        x = nn.ReLU()(x)
        x = x.squeeze(0)
        x = self.transformer(x)
        x = self.batch_norm_2(x)
        x = self.dropout(x)
        out = self.decoder(x)
        return out

model = TransformerModel(input_dim=len(df_train.columns), d_model=64, nhead=8, num_layers=4, dim_feedforward=128).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        
model.apply(init_weights)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

#%%

best_val_loss = float('inf')
save_path = "models/transformers/best_model_cat.pth"

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        X_batch, Y_batch = batch
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            X_batch, Y_batch = batch
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            output = model(X_batch)
            loss = criterion(output, Y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} - train Loss: {train_loss}, val Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at epoch {epoch+1} with val loss: {val_loss}")
    
#%%

save_path = "models/transformers/best_model_cat.pth"
model = TransformerModel(input_dim=len(df_train.columns), d_model=64, nhead=8, num_layers=4, dim_feedforward=128)
model.load_state_dict(torch.load(save_path))
model.eval()
print("Loaded best model for testing")

count_acc(model, X_test, y_test)
evaluate_model_acc(model, X_test, y_test, df_test)

plot_prediction(model, X_test, y_test, df_test)
