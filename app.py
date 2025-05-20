%%writefile app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd # Perlu pandas untuk st.table di metrik
from PIL import Image
from sklearn.cluster import KMeans
from rembg import remove as remove_bg_rembg # Alias untuk menghindari konflik nama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fungsi untuk menghitung Intersection over Union (IoU) / Jaccard Index
def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    if np.sum(union) == 0: # Handle kasus pembagian dengan nol jika union kosong
        return 1.0 if np.sum(intersection) == 0 else 0.0
    return np.sum(intersection) / np.sum(union)

# Fungsi untuk menghapus background menggunakan K-Means
def remove_background_kmeans(image_pil, k=2):
    try:
        # Convert PIL Image to OpenCV format
        image_cv = np.array(image_pil.convert('RGB'))
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Reshape gambar menjadi array piksel
        pixels = image_cv.reshape((-1, 3))
        pixels = np.float32(pixels)

        # Terapkan K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        segmented_image_data = centers[labels.flatten()]
        segmented_image_cv = segmented_image_data.reshape((image_cv.shape))

        # Heuristik untuk menentukan cluster background:
        # Asumsikan cluster yang paling banyak muncul di sudut-sudut adalah background
        corner_pixels = [
            labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1]
        ]
        # Jika gambar sangat kecil, ambil lebih banyak sampel dari border
        h, w, _ = image_cv.shape
        if h < 10 or w < 10: # Gambar sangat kecil
             border_pixels = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
             corner_pixels.extend(border_pixels.flatten().tolist())


        if not corner_pixels: # Jika tidak ada corner pixel (misal 1x1 gambar)
            background_label = labels[0,0] # Ambil saja label pixel pertama
        else:
            background_label = np.argmax(np.bincount(corner_pixels))


        # Buat mask: 0 untuk background, 255 untuk foreground
        mask = np.where(labels.flatten() == background_label, 0, 255).astype(np.uint8)
        mask = mask.reshape((image_cv.shape[0], image_cv.shape[1]))

        # Buat gambar RGBA (dengan alpha channel)
        image_rgba = cv2.cvtColor(image_cv, cv2.COLOR_BGR2BGRA)
        image_rgba[:, :, 3] = mask

        return Image.fromarray(cv2.cvtColor(image_rgba, cv2.COLOR_BGRA2RGBA)), mask
    except Exception as e:
        st.error(f"Error K-Means: {e}")
        # Return original image and an empty mask on error
        empty_mask = np.zeros((np.array(image_pil).shape[0], np.array(image_pil).shape[1]), dtype=np.uint8)
        return image_pil.convert("RGBA"), empty_mask


# Fungsi untuk menghapus background menggunakan rembg (CNN)
def remove_background_cnn(image_pil):
    try:
        # rembg membutuhkan input bytes atau PIL Image
        # dan outputnya adalah PIL Image dengan background transparan
        output_pil = remove_bg_rembg(image_pil)

        # Buat mask dari alpha channel
        if output_pil.mode == 'RGBA':
            alpha = np.array(output_pil.split()[-1])
            mask = np.where(alpha > 128, 255, 0).astype(np.uint8) # Threshold alpha
        else: # Jika output tidak RGBA, asumsikan tidak ada background removal
            st.warning("rembg tidak menghasilkan output RGBA. Tidak dapat membuat mask.")
            mask = np.ones((np.array(output_pil).shape[0], np.array(output_pil).shape[1]), dtype=np.uint8) * 255

        return output_pil, mask
    except Exception as e:
        st.error(f"Error CNN (rembg): {e}")
        empty_mask = np.zeros((np.array(image_pil).shape[0], np.array(image_pil).shape[1]), dtype=np.uint8)
        return image_pil.convert("RGBA"), empty_mask


# Fungsi untuk menghitung metrik evaluasi
def calculate_metrics(true_mask_np, pred_mask_np):
    # Pastikan mask biner (0 atau 1)
    true_mask_binary = (true_mask_np > 128).astype(np.uint8).flatten()
    pred_mask_binary = (pred_mask_np > 128).astype(np.uint8).flatten()

    if len(true_mask_binary) != len(pred_mask_binary):
        return {"error": "Ukuran mask tidak cocok."}
    if len(np.unique(true_mask_binary)) <= 1 or len(np.unique(pred_mask_binary)) <= 1:
        # Hindari error jika salah satu mask hanya punya satu kelas (semua 0 atau semua 1)
        # dalam kasus tertentu precision/recall/f1 bisa tidak terdefinisi
        st.warning("Salah satu mask (ground truth atau prediksi) hanya memiliki satu kelas. Metrik mungkin tidak representatif.")


    metrics = {}
    try:
        metrics['accuracy'] = accuracy_score(true_mask_binary, pred_mask_binary)
    except Exception as e: metrics['accuracy'] = f"Error: {e}"
    try:
        metrics['precision'] = precision_score(true_mask_binary, pred_mask_binary, zero_division=0)
    except Exception as e: metrics['precision'] = f"Error: {e}"
    try:
        metrics['recall'] = recall_score(true_mask_binary, pred_mask_binary, zero_division=0)
    except Exception as e: metrics['recall'] = f"Error: {e}"
    try:
        metrics['f1_score'] = f1_score(true_mask_binary, pred_mask_binary, zero_division=0)
    except Exception as e: metrics['f1_score'] = f"Error: {e}"
    try:
        metrics['iou'] = iou_score(true_mask_binary, pred_mask_binary)
    except Exception as e: metrics['iou'] = f"Error: {e}"

    return metrics

# --- UI Streamlit ---
st.set_page_config(layout="wide", page_title="Background Remover Sederhana")
st.title("🖼️ Aplikasi Penghilang Background Sederhana")
st.markdown("""
Aplikasi ini mendemonstrasikan penghilangan background menggunakan dua metode:
1.  **Segmentasi K-Means:** Mengelompokkan piksel berdasarkan similaritas warna.
2.  **CNN (rembg):** Menggunakan model Deep Learning (U2-Net) yang sudah dilatih.
Metrik evaluasi juga ditampilkan jika *ground truth mask* tersedia.
""")

# Sidebar untuk upload dan pilihan
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded_file = st.file_uploader("Unggah gambar Anda", type=["png", "jpg", "jpeg"])
    
    method = st.radio(
        "Pilih Metode Penghilangan Background:",
        ('Segmentasi K-Means', 'CNN (rembg)'),
        help="K-Means lebih sederhana dan cepat, CNN biasanya lebih akurat tapi lebih lambat."
    )

    if method == 'Segmentasi K-Means':
        k_clusters = st.slider("Jumlah Cluster (K) untuk K-Means:", 2, 5, 2, help="Biasanya 2 cukup untuk foreground/background sederhana.")

    st.markdown("---")
    st.info("Untuk evaluasi, Anda memerlukan 'Ground Truth Mask'. Anda bisa mengunggahnya atau menggunakan contoh.")
    uploaded_mask_file = st.file_uploader("Unggah Ground Truth Mask (opsional)", type=["png", "jpg", "jpeg"])
    
    use_sample_data = st.checkbox("Gunakan Contoh Gambar & Mask untuk Evaluasi", value=False)


# Inisialisasi variabel
original_image_pil = None
ground_truth_mask_pil = None
ground_truth_mask_np = None # Akan diisi nanti jika ada mask

# Load gambar dan mask (jika ada)
if use_sample_data:
    try:
        original_image_pil = Image.open("person.jpg")
        ground_truth_mask_pil = Image.open("person_mask.png").convert('L') # Convert ke Grayscale
        ground_truth_mask_np = np.array(ground_truth_mask_pil)
        st.sidebar.success("Contoh gambar & mask berhasil dimuat.")
    except FileNotFoundError:
        st.sidebar.error("File contoh tidak ditemukan. Pastikan `person.jpg` dan `person_mask.png` ada.")
        original_image_pil = None # Reset jika gagal
        ground_truth_mask_pil = None
        ground_truth_mask_np = None
elif uploaded_file is not None:
    original_image_pil = Image.open(uploaded_file)
    if uploaded_mask_file is not None:
        ground_truth_mask_pil = Image.open(uploaded_mask_file).convert('L')
        ground_truth_mask_np = np.array(ground_truth_mask_pil)

# Tampilkan gambar asli dan mask (jika ada)
col1, col2 = st.columns(2)
with col1:
    if original_image_pil:
        st.image(original_image_pil, caption="Gambar Asli", use_column_width=True)
    else:
        st.info("Silakan unggah gambar atau gunakan contoh data.")

with col2:
    if ground_truth_mask_pil:
        st.image(ground_truth_mask_pil, caption="Ground Truth Mask", use_column_width=True, clamp=True)
    elif original_image_pil: # Hanya tampilkan jika gambar asli ada tapi mask tidak
        st.info("Unggah Ground Truth Mask jika ingin melakukan evaluasi.")

# Tombol proses
if original_image_pil:
    if st.button(f"🚀 Hapus Background dengan {method}", use_container_width=True):
        with st.spinner(f"Memproses dengan {method}..."):
            result_image_pil = None
            pred_mask_np = None

            if method == 'Segmentasi K-Means':
                result_image_pil, pred_mask_np = remove_background_kmeans(original_image_pil, k=k_clusters)
            elif method == 'CNN (rembg)':
                result_image_pil, pred_mask_np = remove_background_cnn(original_image_pil)

            if result_image_pil and pred_mask_np is not None:
                st.subheader("✨ Hasil Penghilangan Background")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.image(result_image_pil, caption=f"Hasil ({method})", use_column_width=True)
                    # Sediakan tombol download
                    from io import BytesIO
                    buf = BytesIO()
                    result_image_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="📥 Unduh Hasil (.png)",
                        data=byte_im,
                        file_name="background_removed_result.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with res_col2:
                    st.image(pred_mask_np, caption="Prediksi Mask", use_column_width=True, clamp=True)


                # --- Evaluasi ---
                if ground_truth_mask_np is not None:
                    st.subheader("📊 Metrik Evaluasi")
                    # Pastikan ukuran mask sama, jika tidak, resize pred_mask_np
                    if ground_truth_mask_np.shape != pred_mask_np.shape:
                        st.warning(f"Ukuran Ground Truth Mask ({ground_truth_mask_np.shape}) dan Prediksi Mask ({pred_mask_np.shape}) berbeda. Mencoba me-resize prediksi mask...")
                        pred_mask_pil_resized = Image.fromarray(pred_mask_np).resize(
                            (ground_truth_mask_np.shape[1], ground_truth_mask_np.shape[0]), # (width, height)
                            Image.NEAREST 
                        )
                        pred_mask_np_resized = np.array(pred_mask_pil_resized)
                        # Periksa apakah resize berhasil membuat ukurannya sama
                        if ground_truth_mask_np.shape == pred_mask_np_resized.shape:
                             pred_mask_for_eval = pred_mask_np_resized
                        else:
                            st.error("Gagal me-resize prediksi mask agar ukurannya sama dengan ground truth. Evaluasi tidak dapat dilakukan.")
                            pred_mask_for_eval = None # Tandai agar evaluasi tidak jalan
                    else:
                        pred_mask_for_eval = pred_mask_np

                    if pred_mask_for_eval is not None:
                        metrics = calculate_metrics(ground_truth_mask_np, pred_mask_for_eval)
                        if "error" in metrics:
                            st.error(metrics["error"])
                        else:
                            st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Nilai']))
                            st.markdown("""
                            **Penjelasan Metrik:**
                            - **Accuracy:** Persentase piksel yang diklasifikasikan dengan benar (foreground/background).
                            - **Precision (Foreground):** Dari semua piksel yang diprediksi sebagai foreground, berapa banyak yang benar.
                            - **Recall (Foreground):** Dari semua piksel foreground yang sebenarnya, berapa banyak yang berhasil diprediksi.
                            - **F1-Score (Foreground):** Rata-rata harmonik dari Precision dan Recall.
                            - **IoU (Intersection over Union):** Luas irisan antara prediksi dan ground truth, dibagi luas gabungannya. Semakin tinggi semakin baik.
                            """)
                    else:
                        st.warning("Tidak dapat melakukan evaluasi karena perbedaan ukuran mask yang tidak teratasi.")

                elif uploaded_file and not uploaded_mask_file and not use_sample_data: # Hanya jika gambar diupload pengguna
                    st.info("Untuk melihat metrik evaluasi, silakan unggah 'Ground Truth Mask' yang sesuai dengan gambar Anda.")
else:
    st.info("Menunggu gambar diunggah atau contoh data dipilih...")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ menggunakan Python, Streamlit, OpenCV, Scikit-learn, dan Rembg.")
