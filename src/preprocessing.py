import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore

def preprocess_full(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # 1) Aggregate scans
    if config['preprocessing']['aggregate']:
        grp = config['training']['group_by']
        spec_cols = df.columns[:112].tolist()
        feat_cols = spec_cols + config['training']['targets']
        df = df.groupby(grp)[feat_cols].mean().reset_index()

    # 2) Extract spectra
    real = df.iloc[:,  len(config['training']['group_by']): len(config['training']['group_by'])+56]
    imag = df.iloc[:,  len(config['training']['group_by'])+56: len(config['training']['group_by'])+112]

    # 3) Magnitude
    mag = np.sqrt(real.values**2 + imag.values**2)
    spec_df = pd.DataFrame(mag, columns=[c.replace('_real','') for c in real.columns])

    # 4) Outlier removal
    y_df = df[config['training']['targets']]
    z = y_df.apply(lambda x: zscore(x, nan_policy='omit'))
    mask = (np.abs(z) <= config['preprocessing']['outlier_removal']['threshold']).all(axis=1)
    spec_df = spec_df.loc[mask].reset_index(drop=True)
    df_targets = y_df.loc[mask].reset_index(drop=True)
    df_group = (df[config['training']['group_by'][0]].astype(str) + '_'
                + df[config['training']['group_by'][1]].astype(str)).loc[mask]

    # 5) Scale
    scaler = RobustScaler() if config['preprocessing']['scaler']=='robust' else StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(spec_df), columns=spec_df.columns)

    # 6) Feature mode
    mode = config['preprocessing']['feature_mode']
    if mode == 'all_bands':
        X_feat = scaled
    elif mode == 'band_selection':
        freqs = [float(f.split('_')[0]) for f in config['preprocessing']['feature_selection']['features']]
        cols = [col for col in scaled.columns if any(abs(float(col.split('_')[0]) - num)<1e-6 for num in freqs)]
        X_feat = scaled[cols]
    elif mode == 'pca_only':
        n_pc = config['preprocessing']['pca']['n_components']
        pca = PCA(n_components=n_pc)
        pcs = pca.fit_transform(scaled.values)
        X_feat = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(n_pc)])
    else:
        raise ValueError(f"Unknown feature_mode: {mode}")

    # assemble
    df_all = pd.concat([X_feat.reset_index(drop=True), df_targets], axis=1)
    df_all['group'] = df_group.values
    return df_all