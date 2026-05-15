import torch
import numpy as np
from dataloaders import NSDdataset
from models import FeatCore
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # <-- aggiunto
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit PCA on ResNet features")
    parser.add_argument("--subject", default=8, type=int)
    parser.add_argument("--roi", default="V1v", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--explained_var", default=0.95, type=float)
    args = parser.parse_args()

    subject = args.subject
    roi = args.roi
    batch_size = args.batch_size
    explained_var = args.explained_var

    # =========================
    # DATA
    # =========================
    dataset = NSDdataset(mode="train", subject=subject, roi=roi)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # =========================
    # MODEL
    # =========================
    core = FeatCore(pretrained=True, finetune=False)
    core.cuda()
    core.eval()

    # =========================
    # FEATURE EXTRACTION
    # =========================
    features = []

    with torch.no_grad():
        for x, _ in tqdm(loader):
            x = x.cuda().float()
            f = core(x)  # (B, C, H, W)
            features.append(f.cpu().numpy())

    features = np.concatenate(features, axis=0)
    print("Feature matrix shape:", features.shape)

    # =========================
    # FEATURE NORMALIZATION
    # =========================
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # standardizza ogni feature a media 0 e varianza 1

    # =========================
    # PCA FIT (95% VARIANCE)
    # =========================
    pca = PCA(
        n_components=explained_var,
        svd_solver="full",
        whiten=False
    )
    pca.fit(features_scaled)

    # =========================
    # SAVE
    # =========================
    out_path = f"pca_S{subject}_{roi}_var{int(explained_var*100)}.npz"
    np.savez(
        out_path,
        components=pca.components_,
        mean=pca.mean_,
        scaler_mean=scaler.mean_,       # <-- salviamo anche i parametri dello scaler
        scaler_scale=scaler.scale_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    print("Saved PCA to:", out_path)
    print("Number of components:", pca.components_.shape[0])
    print("Explained variance (sum):", pca.explained_variance_ratio_.sum())

