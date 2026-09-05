"""Weight resolution and download.

Both stages' weights live on the Hugging Face Hub; nothing is vendored in this
package. Downloads are skipped when the target directory already exists.
"""

import os

DEFAULT_CACHE = os.path.expanduser("~/.cache/vsr")

SR_REPO = "gpanaretou/adcsr"
INTERPOLATION_REPO = "gpanaretou/practical-rife-interpolation"

SR_MODEL_RELPATH = os.path.join("weight", "net_params_200.pkl")
SR_DECODER_RELPATH = os.path.join("weight", "pretrained", "halfDecoder.ckpt")
INTERPOLATION_RELPATH = "train_log"


def _snapshot(repo_id: str, local_dir: str, marker: str) -> str:
    if os.path.isdir(os.path.join(local_dir, marker)):
        return local_dir

    from huggingface_hub import snapshot_download

    print(f"[vsr] downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model")
    return local_dir


def ensure_super_resolution_weights(cache_dir: str = DEFAULT_CACHE) -> tuple[str, str]:
    """Returns (model_ckpt_path, decoder_ckpt_path), downloading if needed."""
    local_dir = os.path.join(cache_dir, "super_resolution")
    _snapshot(SR_REPO, local_dir, marker="weight")

    model_ckpt = os.path.join(local_dir, SR_MODEL_RELPATH)
    decoder_ckpt = os.path.join(local_dir, SR_DECODER_RELPATH)
    for path in (model_ckpt, decoder_ckpt):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"expected weight file missing after download: {path}"
            )
    return model_ckpt, decoder_ckpt


def ensure_interpolation_weights(cache_dir: str = DEFAULT_CACHE) -> str:
    """Returns the RIFE train_log directory, downloading if needed."""
    local_dir = os.path.join(cache_dir, "interpolation")
    _snapshot(INTERPOLATION_REPO, local_dir, marker=INTERPOLATION_RELPATH)

    train_log = os.path.join(local_dir, INTERPOLATION_RELPATH)
    if not os.path.isfile(os.path.join(train_log, "flownet.pkl")):
        raise FileNotFoundError(f"flownet.pkl missing under {train_log}")
    return train_log
