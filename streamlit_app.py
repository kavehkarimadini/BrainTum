import streamlit as st
import numpy as np
import h5py
import requests
import matplotlib.pyplot as plt

# -----------------------------
# Hugging Face API configuration
# -----------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/YOUR_USERNAME/YOUR_MODEL"
HF_TOKEN = "YOUR_HF_TOKEN"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# -----------------------------
# Dice score
# -----------------------------
def dice_score(pred, gt, num_classes=4):
    scores = []

    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)

        intersection = np.sum(pred_c * gt_c)
        union = np.sum(pred_c) + np.sum(gt_c)

        if union == 0:
            scores.append(1.0)
        else:
            scores.append((2 * intersection) / union)

    return scores


# -----------------------------
# HuggingFace inference
# -----------------------------
def query_hf_api(image_tensor):
    """Call HuggingFace API; expects response with 'mask' and optional 'tumor_prob'."""
    response = requests.post(
        "https://kavehkarimadini-braintum-api.hf.space/predict",
        json={"image": image_tensor.tolist()},
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"API returned status {response.status_code}. "
            f"Response: {response.text[:500] if response.text else '(empty)'}"
        )

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError(
            "API did not return valid JSON. "
            f"Response (first 500 chars): {response.text[:500] if response.text else '(empty)'}"
        )

    if "mask" not in data:
        raise RuntimeError(f"API response missing 'mask'. Keys: {list(data.keys())}")

    pred = np.array(data["mask"])
    tumor_prob = data.get("tumor_prob")  # None if not returned (no gating)
    return pred, tumor_prob


# -----------------------------
# BraTSDataset-style preprocessing (image + mask)
# -----------------------------
def preprocess_mask(mask):
    """
    Convert BraTS mask to contiguous class labels, matching BraTSDataset.__getitem__.
    - One-hot (H, W, 3): channels 0,1,2 → classes 1,2,3 (necrotic, edema, enhancing).
    - Single-channel (H, W) with values {0,1,2,4}: map 4→3.
    Returns (H, W) int64 with values in {0, 1, 2, 3}.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        # One-hot: (H, W, 3)
        new_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        new_mask[mask[:, :, 0] == 1] = 1   # Necrotic tumor
        new_mask[mask[:, :, 1] == 1] = 2   # Edema
        new_mask[mask[:, :, 2] == 1] = 3   # Enhancing tumor
        return new_mask
    else:
        # Single-channel (0, 1, 2, 4)
        mask_mapped = np.zeros(mask.shape, dtype=np.int64)
        mask_mapped[mask == 1] = 1
        mask_mapped[mask == 2] = 2
        mask_mapped[mask == 4] = 3
        return mask_mapped


def preprocess_image_for_model(image):
    """
    Match BraTSDataset: HWC -> CHW, float. No normalization (dataset does not normalize).
    Returns (C, H, W) float32.
    """
    image = np.asarray(image, dtype=np.float32)
    # HWC -> CHW (same as dataset)
    image = np.transpose(image, (2, 0, 1))
    return image


def postprocess_prediction(pred, tumor_prob=None, target_shape=None):
    """
    Match postprocessing from predict_and_show:
    - Convert logits to class indices (argmax on class dim)
    - If tumor_prob < 0.5, force prediction to background (all zeros)
    Returns 2D numpy array (H, W) with values in {0, 1, 2, 3}.
    """
    pred = np.array(pred)
    # Reduce to 2D class indices (argmax over classes if logits)
    if pred.ndim == 4:
        if pred.shape[1] <= 4 and pred.shape[1] < pred.shape[-1]:
            pred = np.argmax(pred, axis=1)[0]
        else:
            pred = np.argmax(pred, axis=-1)[0]
    elif pred.ndim == 3:
        if pred.shape[0] == 1 and pred.shape[-1] not in (1, 2, 3, 4):
            pred = pred[0]
        elif pred.shape[-1] in (1, 2, 3, 4):
            pred = np.argmax(pred, axis=-1)
            if pred.shape[0] == 1:
                pred = pred[0]
        else:
            pred = np.argmax(pred, axis=0)
    elif pred.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected pred shape: {pred.shape}")

    pred = np.squeeze(pred).astype(np.int64)

    # Gate by tumor probability (same as predict_and_show)
    if tumor_prob is not None and tumor_prob < 0.5:
        pred = np.zeros_like(pred)

    if target_shape is not None and pred.shape != target_shape:
        if pred.size == np.prod(target_shape):
            pred = np.reshape(pred, target_shape)
        else:
            raise ValueError(
                f"Prediction shape {pred.shape} cannot be aligned with target {target_shape}"
            )
    return pred


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("BrainTum Inference (HuggingFace Backend)")

uploaded = st.file_uploader("Upload .h5 slice", type=["h5"])

if uploaded:

    with h5py.File(uploaded, "r") as f:
        image = f["image"][:]
        mask = f["mask"][:]

    # Display: raw H5 data (no preprocessing)
    image_display = np.asarray(image)
    # if mask.ndim == 3:
    #     mask_display = mask[:,:,0]  # first channel for display
    # else:
    mask_display = np.asarray(mask)

    st.subheader("Original MRI & Ground Truth")

    col1, col2 = st.columns(2)

    with col1:
        # st.image(image_display, caption="MRI Slice (FLAIR)",channels="gray",clamp=True)
        # Display with same colormap as predict_and_show (gnuplot, vmin=0, vmax=3)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image_display[:,:,0], cmap="gray", vmin=0, vmax=3)
        ax.set_title("MRI Slice (FLAIR)")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        # Raw mask: use matplotlib so we can set vmax=4 for single-channel (0,1,2,4)
        fig_m, ax_m = plt.subplots(figsize=(5, 5))
        ax_m.imshow(preprocess_mask(mask_display), cmap="gnuplot", vmin=0, vmax=3)
        ax_m.set_title("Ground Truth Mask")
        ax_m.axis("off")
        plt.tight_layout()
        st.pyplot(fig_m)
        plt.close(fig_m)

    if st.button("Run Inference on HuggingFace"):
        # Preprocessing only for inference: BraTSDataset-style image + mask
        image_chw = preprocess_image_for_model(image)
        image_tensor = np.expand_dims(image_chw, 0)  # (1, C, H, W) for API
        mask_mapped = preprocess_mask(mask)  # for Dice and target_shape only

        try:
            raw_pred, tumor_prob = query_hf_api(image_tensor)
        except (RuntimeError, requests.exceptions.RequestException) as e:
            st.error(f"Inference failed: {e}")
            st.stop()

        gt_2d = np.squeeze(mask_mapped)
        if gt_2d.ndim == 3:
            gt_2d = gt_2d[:, :, 0]

        # Postprocess like predict_and_show: argmax + tumor gate
        pred_mask = postprocess_prediction(
            raw_pred, tumor_prob=tumor_prob, target_shape=gt_2d.shape
        )

        dice = dice_score(pred_mask, gt_2d)

        # Caption matching predict_and_show
        if tumor_prob is not None:
            if tumor_prob < 0.5:
                title_pred = f"Prediction (Gated: No tumor, p={tumor_prob:.2f})"
            else:
                title_pred = f"Prediction (Tumor detected, p={tumor_prob:.2f})"
        else:
            title_pred = "Prediction"

        st.subheader("Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(image_display[:,:,0], cmap="gray", vmin=0, vmax=3)
            ax.set_title("MRI Slice (FLAIR)")
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig_m, ax_m = plt.subplots(figsize=(5, 5))
            ax_m.imshow(preprocess_mask(mask_display), cmap="gnuplot", vmin=0, vmax=3)
            ax_m.set_title("Ground Truth Mask")
            ax_m.axis("off")
            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close(fig_m)

        with col3:
            # Display with same colormap as predict_and_show (gnuplot, vmin=0, vmax=3)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(pred_mask, cmap="gnuplot", vmin=0, vmax=3)
            ax.set_title(title_pred)
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.write("Dice per class:", dice)
        st.write("Mean tumor dice:", np.mean(dice[1:]))