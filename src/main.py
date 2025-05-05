import argparse
import yaml
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from data_loader import load_raw
from preprocessing import preprocess_full
from models.random_forrest import build_rf
from models.oned_cnn import OneDCNN
from utils import eval_metrics, save_model
import os
from utils import save_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', nargs='+', help='Targets to predict')
    parser.add_argument('--model', choices=['random_forest','oned_cnn'], help='Model choice override')
    args = parser.parse_args()

    # Load config
    config = yaml.safe_load(open('config/config.yaml'))
    # Override targets/model if passed
    if args.targets:
        config['training']['targets'] = args.targets
    if args.model:
        config['model']['choice'] = args.model

    # 1) Load and preprocess
    df = load_raw(config['data']['raw_path'])
    df_all = preprocess_full(df, config)

    targets = config['training']['targets']
    X = df_all.drop(columns=targets + ['group']).values
    y = df_all[targets].values
    groups = df_all['group'].values

    # 2) Train/validation split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config['preprocessing']['split']['validation_split'],
        random_state=config['preprocessing']['split']['random_state']
    )
    train_idx, val_idx = next(splitter.split(X, y, groups))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx],   y[val_idx]

    # 3) Model training
    choice = config['model']['choice']
    if choice == 'random_forest':
        model = build_rf(config)
        # no internal CV; train on full train split
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        print("Validation metrics:", eval_metrics(y_val, preds))
        os.makedirs('models', exist_ok=True)
        save_pickle(model, 'models/random_forest.pkl')
        print("→ Random Forest saved to models/random_forest.pkl")

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X_train.shape[1]
        model = OneDCNN(config, input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['oned_cnn']['lr'])
        loss_fn = nn.MSELoss()
        batch_size = config['model']['oned_cnn']['batch_size']

        # DataLoader
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(config['model']['oned_cnn']['epochs']):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = loss_fn(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(train_ds)
            print(f"Epoch {epoch+1}/{config['model']['oned_cnn']['epochs']} - Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
            preds = model(Xv).cpu().numpy()
        print("Validation metrics:", eval_metrics(y_val, preds))
        save_model(model, 'models/oned_cnn.pt')

if __name__ == '__main__':
    main()