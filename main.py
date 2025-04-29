from datasets.precipitation_dataset import PrecipitationDataset
from models.hydro_fusion_net import HydroFusionNet
from training.train import train_model
from training.evaluate import evaluate_model, save_classification_results
from utils.preprocessing import preprocess_data
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config = json.load(open("config/info.json"))
data_dir = config["data_dir"]

# Load data
df = pd.read_pickle(data_dir)
df = preprocess_data(df)  # Downsampling, feature engineering, etc.

# Split
train_df, val_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["bin"]
)

# Scaling
feature_scaler = StandardScaler()
diff_scaler = StandardScaler()
train_dataset = PrecipitationDataset(
    train_df, feature_scaler, diff_scaler, is_train=True
)
val_dataset = PrecipitationDataset(val_df, feature_scaler, diff_scaler, is_train=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Model
model = HydroFusionNet()

# Train
train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device=device)

# Evaluate
evaluate_model(model, val_loader, device=device)
save_classification_results(
    model, val_loader, results_dir="classifierResults", device=device
)
