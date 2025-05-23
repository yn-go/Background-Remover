#%%writefile app.py
import streamlit as st
from PIL import Image
import numpy as np
from rembg import remove
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# Fungsi untuk menghitung metrik evaluasi
def calculate_metrics(true_mask_pil, pred_mask_pil):
    """
    Menghitung metrik evaluasi antara true mask dan predicted mask.
    Kedua mask harus berupa PIL Image grayscale (mode 'L') dan biner (0 atau 255).
    """
    # Pastikan ukuran mask sama, jika tidak, resize predicted mask
    if true_mask_pil.size != pred_mask_pil.size:
        st.warning(f"Ukuran ground truth ({true_mask_pil.size}) dan prediksi ({pred_mask_pil.size}) berbeda. Prediksi di-resize.")
        pred_mask_pil = pred_mask_pil.resize(true_mask_pil.size, Image.Resampling.NEAREST)

    true_mask_np = np.array(true_mask_pil).astype(np.uint8)
    pred_mask_np = np.array(pred_mask_pil).astype(np.uint8)

    # Binarisasi (pastikan 0 untuk background, 1 untuk foreground)
    # Asumsi nilai > 127 adalah foreground (255), lainnya background (0)
    true_mask_np_binary = (true_mask_np > 127).astype(np.uint8)
    pred_mask_np_binary = (pred_mask_np > 127).astype(np.uint8)

    # Flatten array untuk kalkulasi
    y_true = true_mask_np_binary.flatten()
    y_pred = pred_mask_np_binary.flatten()

    metrics = {}
    try:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['IoU (Jaccard)'] = jaccard_score(y_true, y_pred, zero_division=0) # Intersection over Union
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return None

    return metrics

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Penghilang Background Otomatis")

st.title("✂️ Aplikasi Penghilang Background Gambar")
st.markdown("""
Aplikasi ini menggunakan model AI (U2-Net via `rembg`) untuk secara otomatis menghilangkan background dari gambar.
Anda juga dapat mengunggah *ground truth mask* untuk mengevaluasi hasilnya.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("🖼️ Unggah Gambar")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["png", "jpg", "jpeg"])
    ground_truth_file = st.file_uploader(
        "Unggah Ground Truth Mask (opsional, .png format, hitam-putih)",
        type=["png"],
        help="Masker harus hitam (background) dan putih (foreground)."
    )

original_image_pil = None
ground_truth_pil = None

if uploaded_file is not None:
    try:
        original_image_pil = Image.open(uploaded_file).convert("RGB") # Pastikan RGB
        with col1:
            st.image(original_image_pil, caption="Gambar Asli", use_column_width=True)
    except Exception as e:
        st.error(f"Error memuat gambar asli: {e}")
        original_image_pil = None # Reset jika error

if ground_truth_file is not None and original_image_pil: # Hanya proses GT jika gambar asli ada
    try:
        ground_truth_pil = Image.open(ground_truth_file).convert("L") # Convert ke Grayscale (mask)
        with col1:
            st.image(ground_truth_pil, caption="Ground Truth Mask", use_column_width=True)
    except Exception as e:
        st.error(f"Error memuat ground truth mask: {e}")
        ground_truth_pil = None # Reset jika error
elif ground_truth_file is not None and not original_image_pil:
    st.warning("Silakan unggah gambar asli terlebih dahulu sebelum mengunggah ground truth.")


with col2:
    st.header("✨ Hasil & Evaluasi")
    if original_image_pil:
        if st.button("🚀 Hapus Background!", type="primary", use_container_width=True):
            with st.spinner("Sedang memproses gambar... Ini mungkin memakan waktu beberapa detik."):
                try:
                    # Convert PIL Image to bytes
                    img_byte_arr = io.BytesIO()
                    original_image_pil.save(img_byte_arr, format='PNG') # Format PNG agar bisa handle alpha
                    img_byte_arr = img_byte_arr.getvalue()

                    # Hapus background menggunakan rembg
                    result_bytes = remove(img_byte_arr)
                    result_image_pil = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

                    st.image(result_image_pil, caption="Gambar Tanpa Background", use_column_width=True)

                    # Tombol download
                    buf = io.BytesIO()
                    result_image_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="📥 Unduh Hasil (.png)",
                        data=byte_im,
                        file_name=f"bg_removed_{uploaded_file.name.split('.')[0]}.png",
                        mime="image/png",
                        use_container_width=True
                    )

                    # --- Evaluasi ---
                    if ground_truth_pil:
                        st.subheader("📊 Metrik Evaluasi")
                        # Dapatkan alpha channel sebagai predicted mask dari hasil rembg
                        # Alpha channel adalah channel ke-4 pada gambar RGBA
                        if result_image_pil.mode == 'RGBA':
                            pred_mask_pil = result_image_pil.split()[-1] # Ambil alpha channel
                        else: # Jika output bukan RGBA (jarang terjadi dengan rembg)
                            st.warning("Hasil tidak memiliki alpha channel untuk evaluasi.")
                            pred_mask_pil = Image.new('L', result_image_pil.size, 0) # Mask kosong

                        metrics = calculate_metrics(ground_truth_pil, pred_mask_pil)
                        if metrics:
                            for metric_name, metric_value in metrics.items():
                                st.metric(label=metric_name, value=f"{metric_value:.4f}")
                        else:
                            st.error("Gagal menghitung metrik.")
                    else:
                        st.info("ℹ️ Unggah Ground Truth Mask untuk melihat metrik evaluasi.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
        else:
            st.info("Klik tombol 'Hapus Background!' setelah mengunggah gambar.")
    else:
        st.info("Silakan unggah gambar terlebih dahulu di kolom sebelah kiri.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ menggunakan Python, Streamlit, dan `rembg`.")
