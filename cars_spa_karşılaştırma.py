import os
import spectral.io.envi as envi
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

def load_all_images(main_folder_path):
    day_folders = [os.path.join(main_folder_path, day_folder) for day_folder in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, day_folder))]
    img_datas = []
    file_list = []
    for folder_path in day_folders:
        day_file_list = os.listdir(folder_path)
        for hdr_file in day_file_list:
            if hdr_file.endswith('.hdr'):
                hdr_path = os.path.join(folder_path, hdr_file)
                bin_file = hdr_file.replace('.hdr', '.bin')
                bin_path = os.path.join(folder_path, bin_file)
                img_hdr = envi.open(hdr_path, image=bin_path)
                img_data = img_hdr.load()
                img_datas.append(img_data)
                file_list.append(hdr_file)
    return img_datas, file_list

def successive_projections_algorithm(X, num_features):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    _, n_features = X.shape
    selected_features = []
    remaining_features = list(range(n_features))

    # Step 1: Initialize with the feature having maximum variance
    variances = np.var(X, axis=0)
    first_feature = np.argmax(variances)
    selected_features.append(first_feature)
    remaining_features.remove(first_feature)

    # Step 2: Iteratively select features based on maximum projection
    for _ in range(1, num_features):
        max_projection = -np.inf
        max_feature = None

        for feature in remaining_features:
            projection = np.sum(np.abs(np.dot(X[:, feature], X[:, selected_features])))
            if projection > max_projection:
                max_projection = projection
                max_feature = feature

        selected_features.append(max_feature)
        remaining_features.remove(max_feature)

    return selected_features

def competitive_adaptive_reweighted_sampling(X, num_features):
    pca = PCA(n_components=num_features)
    pca.fit(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    selected_features = np.argsort(explained_variance_ratio)[::-1][:num_features]
    return selected_features

# Hiperspektral verileri yükleme
main_folder_path = 'VIS_COR'  # Veri seti klasörü
img_datas, file_list = load_all_images(main_folder_path)


# Örnekleme büyüklüğü
sample_size = 100


np.random.seed(0)  # Tekrarlanabilirlik için
sample_indices = np.random.choice(len(img_datas), sample_size, replace=False)
sampled_img_datas = [img_datas[i] for i in sample_indices]

# Tüm örneklemleri birleştirerek yeniden şekillendirme tek veri matrisi haline getirdim
X_combined = np.vstack([img.reshape(-1, img.shape[-1]) for img in sampled_img_datas])

# SPA ile bant seçimi süresi ölçümü
start_time = time.time()
num_features_sp = 30  # Seçilecek bant sayısı SPA için
selected_bands_indices_sp = successive_projections_algorithm(X_combined, num_features_sp)
spa_execution_time = time.time() - start_time

# CARS ile bant seçimi süresi ölçümü
start_time = time.time()
num_features_cars = 30  # Seçilecek bant sayısı CARS için
selected_bands_indices_cars = competitive_adaptive_reweighted_sampling(X_combined, num_features_cars)
cars_execution_time = time.time() - start_time

# Süreleri yazdırma
print(f"SPA algoritması çalışma süresi: {spa_execution_time:.2f} saniye")
print(f"CARS algoritması çalışma süresi: {cars_execution_time:.2f} saniye")

# Seçilen bantları her bir görüntü için yeniden şekillendirme
selected_bands_sp = [img[:, :, selected_bands_indices_sp] for img in sampled_img_datas]
selected_bands_cars = [img[:, :, selected_bands_indices_cars] for img in sampled_img_datas]

# Örnek olarak ilk görüntünün boyutları
print(f"İlk örneklemede SPA ile seçilen bantlar boyutları: {selected_bands_sp[0].shape}")
print(f"İlk örneklemede CARS ile seçilen bantlar boyutları: {selected_bands_cars[0].shape}")