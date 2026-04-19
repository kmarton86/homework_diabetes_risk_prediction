# analysis.py

def dataset_summary(X, y):
    return {
        "shape": X.shape,
        "feature_means": X.mean().to_dict(),
        "feature_std": X.std().to_dict(),
        "target_mean": float(y.mean()),
        "target_min": float(y.min()),
        "target_max": float(y.max())
    }


def class_distribution(y, threshold):
    labels = (y > threshold).astype(int)

    return {
        "not_endangered": int((labels == 0).sum()),
        "endangered": int((labels == 1).sum())
    }