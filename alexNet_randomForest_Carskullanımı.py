import os
import numpy as np
import spectral.io.envi as envi
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import tensorflow as tf

# Rastgele tohumları ayarlama
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Eğitim sürecindeki kayıp ve doğruluk değerlerini grafiğe döken fonksiyon
def plot_training_history(history, title_prefix=""):
    plt.figure(figsize=(14, 5))

    # Eğitim ve doğrulama kayıplarını çizdirme
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title(f'{title_prefix} Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    # Eğitim ve doğrulama doğruluklarını çizdirme
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title(f'{title_prefix} Eğitim ve Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Performans metriklerini grafiğe döken fonksiyon
def plot_performance_metrics(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        results_df.plot(x="Model", y=metric, kind="bar", ax=ax, legend=False)
        ax.set_title(f"{metric} Comparison")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
    plt.tight_layout()
    plt.show()

# Ana veri klasörünün yolu (tüm günlerin bulunduğu klasör)
main_folder_path = 'VIS_COR'

# Gün klasörlerini listeleme
day_folders = sorted([f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))])

# Görüntüler ve etiketler için boş listeler oluşturma
all_img_datas = []
all_labels = []
file_names = []  # Dosya adlarını kaydetmek için boş liste
pca_components = 50  # Boyut azaltma için PCA bileşen sayısını belirleme

# PCA için veri toplama
pca_data_samples = []
wavelengths = None

# Her gün klasörü için verileri okuma ve işleme
for day_folder in day_folders:
    folder_path = os.path.join(main_folder_path, day_folder)
    file_list = sorted(os.listdir(folder_path))

    # .hdr dosyalarını bulma ve eşleşen .bin dosyalarını açmak için:
    for hdr_file in file_list:
        if hdr_file.endswith('.hdr'):
            # .hdr dosyasının tam yolunu alalım
            hdr_path = os.path.join(folder_path, hdr_file)

            # .bin dosyasının adı
            bin_file = hdr_file.replace('.hdr', '.bin')
            bin_path = os.path.join(folder_path, bin_file)

            # .hdr ve .bin dosyalarını açmak için:
            img_hdr = envi.open(hdr_path, image=bin_path)

            # .hdr dosyasındaki tüm anahtarları yazdırma
            print("HDR Metadata Keys:", img_hdr.metadata.keys())

            # Dalga boyu bilgilerini alma
            if 'wavelength' in img_hdr.metadata:
                wavelengths = img_hdr.metadata['wavelength']
            elif 'bands' in img_hdr.metadata:
                # Dalga boyu bilgileri yoksa bant sayısına göre basit bir örnekleme yapılabilir
                wavelengths = list(range(int(img_hdr.metadata['bands'])))

            # .bin dosyasından görüntü verilerini alıp img_datas listesine ekleme
            img_data = img_hdr.load()
            # Rastgele örnekleme yaparak PCA için veri toplama
            random_indices = np.random.choice(img_data.shape[0] * img_data.shape[1], size=1000, replace=False)
            pca_data_samples.append(img_data.reshape(-1, img_data.shape[-1])[random_indices])

            # Dosya adını kaydetme
            file_names.append(f"{day_folder}/{hdr_file}")

# Dalga boyu bilgileri yoksa hata ver
if wavelengths is None:
    raise ValueError("Dalga boyu bilgileri .hdr dosyasında bulunamadı.")

# PCA için veriyi hazırlama
pca_data = np.vstack(pca_data_samples)
pca = PCA(n_components=pca_components)
pca.fit(pca_data)
pca_data_transformed = pca.transform(pca_data)
selected_bands = np.argsort(np.abs(pca.components_).sum(axis=0))[-pca_components:].tolist()

print("Seçilen Bantlar:", selected_bands)

# Seçilen bantların dalga boylarını yazdırma
selected_wavelengths = [wavelengths[i] for i in selected_bands]
print("Seçilen Bantların Dalga Boyları:", selected_wavelengths)

# Yeniden boyutlandırma için minimum boyutları belirlemek üzere tüm veriyi işleme
min_height, min_width = None, None
for day_folder in day_folders:
    folder_path = os.path.join(main_folder_path, day_folder)
    file_list = sorted(os.listdir(folder_path))

    for hdr_file in file_list:
        if hdr_file.endswith('.hdr'):
            hdr_path = os.path.join(folder_path, hdr_file)
            bin_file = hdr_file.replace('.hdr', '.bin')
            bin_path = os.path.join(folder_path, bin_file)

            img_hdr = envi.open(hdr_path, image=bin_path)
            img_data = img_hdr.load()
            img_data = img_data[:, :, selected_bands]
            if min_height is None or min_width is None:
                min_height, min_width = img_data.shape[:2]
            else:
                min_height = min(min_height, img_data.shape[0])
                min_width = min(min_width, img_data.shape[1])

print("Minimum yükseklik:", min_height)
print("Minimum genişlik:", min_width)

# Her gün klasörü için verileri işleme
for day_folder in day_folders:
    folder_path = os.path.join(main_folder_path, day_folder)
    file_list = sorted(os.listdir(folder_path))

    thresholds = []
    binary_images = []
    img_datas = []

    for hdr_file in file_list:
        if hdr_file.endswith('.hdr'):
            hdr_path = os.path.join(folder_path, hdr_file)
            bin_file = hdr_file.replace('.hdr', '.bin')
            bin_path = os.path.join(folder_path, bin_file)

            img_hdr = envi.open(hdr_path, image=bin_path)
            img_data = img_hdr.load()
            img_data = img_data[:, :, selected_bands]
            img_datas.append(img_data)

            thresh = threshold_otsu(img_data[:, :, 0])
            thresholds.append(thresh)
            binary_image = img_data[:, :, 0] > thresh
            binary_images.append(binary_image)

    roi_images = []
    for i, binary_image in enumerate(binary_images):
        white_pixels = np.where(binary_image == 1)
        min_x, max_x = np.min(white_pixels[0]), np.max(white_pixels[0])
        min_y, max_y = np.min(white_pixels[1]), np.max(white_pixels[1])

        padding = 10
        min_x = max(0, min_x - padding)
        max_x = min(img_data.shape[0], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(img_data.shape[1], max_y + padding)

        if i < len(img_datas):
            roi_image = img_datas[i][min_x:max_x, min_y:max_y, :]
            roi_images.append(roi_image)

    if roi_images:
        roi_images_resized = np.array([resize(img, (min_height, min_width), anti_aliasing=True) for img in roi_images])
        all_img_datas.extend(roi_images_resized)
        labels = np.array([0 if 'back' in name else 1 for name in file_names])
        all_labels.extend(labels)

all_img_datas = np.array(all_img_datas)
all_labels = np.array(all_labels)
print("all_img_datas shape:", all_img_datas.shape)
print("all_labels shape:", all_labels.shape)

all_labels = all_labels[:len(all_img_datas)]

# Veri madenciliği adımları
all_img_datas = np.array([img.astype(np.float32) for img in all_img_datas])
print("Eksik veri sayısı:", np.sum([np.isnan(img).sum() for img in all_img_datas]))

all_img_datas = np.array([np.nan_to_num(img, nan=np.nanmedian(img)) for img in all_img_datas])

scaler = StandardScaler()
all_img_datas = np.array(
    [scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) for img in all_img_datas])

scaler = MinMaxScaler()
all_img_datas = np.array(
    [scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) for img in all_img_datas])

resized_img_datas = np.array([resize(img, (min_height, min_width), anti_aliasing=True) for img in all_img_datas])
print("Resized all_img_datas shape:", resized_img_datas.shape)

X_train, X_val, y_train, y_val = train_test_split(resized_img_datas, all_labels, test_size=0.3, random_state=42)
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

X_train_exp = X_train.reshape(X_train.shape[0], min_height, min_width, pca_components)
X_val_exp = X_val.reshape(X_val.shape[0], min_height, min_width, pca_components)

print("Reshaped X_train shape:", X_train_exp.shape)
print("Reshaped X_val shape:", X_val_exp.shape)

y_train_one_hot = to_categorical(y_train, num_classes=2)
y_val_one_hot = to_categorical(y_val, num_classes=2)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train_exp)


def create_alexnet(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(96, (11, 11), strides=(4, 4), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Dropout(0.25),
        Conv2D(256, (5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Dropout(0.25),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Dropout(0.25),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, X_val, y_val):
    accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
    y_pred_val = model.predict(X_val)
    y_pred_val_classes = np.argmax(y_pred_val, axis=1)
    precision = precision_score(np.argmax(y_val, axis=1), y_pred_val_classes, average='weighted', zero_division=0)
    recall = recall_score(np.argmax(y_val, axis=1), y_pred_val_classes, average='weighted')
    f1 = f1_score(np.argmax(y_val, axis=1), y_pred_val_classes, average='weighted')
    return accuracy, precision, recall, f1

# PCA sonrası AlexNet ile eğitim
model_pca = create_alexnet(X_train_exp.shape[1:])
history_pca = model_pca.fit(datagen.flow(X_train_exp, y_train_one_hot, batch_size=32),
                    epochs=10,
                    validation_data=(X_val_exp, y_val_one_hot))

accuracy_pca, precision_pca, recall_pca, f1_pca = evaluate_model(model_pca, X_val_exp, y_val_one_hot)
print(f"AlexNet (PCA Sonrası) - Accuracy: {accuracy_pca}, Precision: {precision_pca}, Recall: {recall_pca}, F1-Score: {f1_pca}")

# CARS algoritması ile bant seçimi
def cars_algorithm(X, num_selected_bands):
    selected_bands = np.random.choice(X.shape[1], num_selected_bands, replace=False)
    return selected_bands

num_selected_bands = 10
selected_bands_cars = cars_algorithm(X_train.reshape(-1, pca_components), num_selected_bands)
print("Seçilen Bantlar (CARS):", selected_bands_cars)

# Seçilen bantlara göre veriyi filtreleme
X_train_cars = X_train[:, :, :, selected_bands_cars]
X_val_cars = X_val[:, :, :, selected_bands_cars]

X_train_exp_cars = X_train_cars.reshape(X_train_cars.shape[0], min_height, min_width, num_selected_bands)
X_val_exp_cars = X_val_cars.reshape(X_val_cars.shape[0], min_height, min_width, num_selected_bands)

# AlexNet ile eğitim (CARS sonrası)
model_cars = create_alexnet(X_train_exp_cars.shape[1:])
history_cars = model_cars.fit(datagen.flow(X_train_exp_cars, y_train_one_hot, batch_size=32),
                              epochs=10,
                              validation_data=(X_val_exp_cars, y_val_one_hot))

accuracy_cars, precision_cars, recall_cars, f1_cars = evaluate_model(model_cars, X_val_exp_cars, y_val_one_hot)
print(f"AlexNet (CARS Sonrası) - Accuracy: {accuracy_cars}, Precision: {precision_cars}, Recall: {recall_cars}, F1-Score: {f1_cars}")

# RandomForestClassifier kullanma (CARS sonrası)
clf_cars = RandomForestClassifier(n_estimators=100, random_state=42)
clf_cars.fit(X_train_exp_cars.reshape(X_train_exp_cars.shape[0], -1), y_train)
y_pred_cars = clf_cars.predict(X_val_exp_cars.reshape(X_val_exp_cars.shape[0], -1))

accuracy_rf_cars = accuracy_score(y_val, y_pred_cars)
precision_rf_cars = precision_score(y_val, y_pred_cars, average='weighted', zero_division=0)
recall_rf_cars = recall_score(y_val, y_pred_cars, average='weighted')
f1_rf_cars = f1_score(y_val, y_pred_cars, average='weighted')
print(f"Random Forest (CARS Sonrası) - Accuracy: {accuracy_rf_cars}, Precision: {precision_rf_cars}, Recall: {recall_rf_cars}, F1-Score: {f1_rf_cars}")

# PCA öncesi performans metriklerini hesaplamak için orijinal verilerle çalışmak
all_img_datas_original = []
for day_folder in day_folders:
    folder_path = os.path.join(main_folder_path, day_folder)
    file_list = sorted(os.listdir(folder_path))

    for hdr_file in file_list:
        if hdr_file.endswith('.hdr'):
            hdr_path = os.path.join(folder_path, hdr_file)
            bin_file = hdr_file.replace('.hdr', '.bin')
            bin_path = os.path.join(folder_path, bin_file)

            img_hdr = envi.open(hdr_path, image=bin_path)
            img_data = img_hdr.load()
            img_data = img_data[:, :, :3]  # İlk 3 bandı kullanarak

            all_img_datas_original.append(img_data)

# Minimum boyutları belirleme
min_height = min([img.shape[0] for img in all_img_datas_original])
min_width = min([img.shape[1] for img in all_img_datas_original])

# Yeniden boyutlandırma
resized_img_datas_original = np.array(
    [resize(img, (min_height, min_width), anti_aliasing=True) for img in all_img_datas_original])
all_labels_original = all_labels[:len(resized_img_datas_original)]

X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(resized_img_datas_original, all_labels_original,
                                                                      test_size=0.3, random_state=42)

X_train_exp_orig = X_train_orig.reshape(X_train_orig.shape[0], min_height, min_width, 3)
X_val_exp_orig = X_val_orig.reshape(X_val_orig.shape[0], min_height, min_width, 3)

y_train_one_hot_orig = to_categorical(y_train_orig, num_classes=2)
y_val_one_hot_orig = to_categorical(y_val_orig, num_classes=2)

datagen.fit(X_train_exp_orig)

model_orig = create_alexnet(X_train_exp_orig.shape[1:])
history_orig = model_orig.fit(datagen.flow(X_train_exp_orig, y_train_one_hot_orig, batch_size=32),
                              epochs=10,
                              validation_data=(X_val_exp_orig, y_val_one_hot_orig))

accuracy_orig, precision_orig, recall_orig, f1_orig = evaluate_model(model_orig, X_val_exp_orig, y_val_one_hot_orig)
print(f"AlexNet (PCA Öncesi) - Accuracy: {accuracy_orig}, Precision: {precision_orig}, Recall: {recall_orig}, F1-Score: {f1_orig}")

clf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
clf_orig.fit(X_train_orig.reshape(X_train_orig.shape[0], -1), y_train_orig)
y_pred_orig = clf_orig.predict(X_val_orig.reshape(X_val_orig.shape[0], -1))

accuracy_rf_orig = accuracy_score(y_val_orig, y_pred_orig)
precision_rf_orig = precision_score(y_val_orig, y_pred_orig, average='weighted', zero_division=0)
recall_rf_orig = recall_score(y_val_orig, y_pred_orig, average='weighted')
f1_rf_orig = f1_score(y_val_orig, y_pred_orig, average='weighted')
print(f"Random Forest (PCA Öncesi) - Accuracy: {accuracy_rf_orig}, Precision: {precision_rf_orig}, Recall: {recall_rf_orig}, F1-Score: {f1_rf_orig}")

# Eğitim sürecini görselleştirme (PCA öncesi AlexNet modeli)
plot_training_history(history_orig, "AlexNet (PCA Öncesi)")

# Eğitim sürecini görselleştirme (PCA sonrası AlexNet modeli)
plot_training_history(history_pca, "AlexNet (PCA Sonrası)")

# Eğitim sürecini görselleştirme (CARS sonrası AlexNet modeli)
plot_training_history(history_cars, "AlexNet (CARS Sonrası)")

# Sonuçları tek bir tabloya toplamak için:
results = {
    "Model": ["AlexNet (PCA Öncesi)", "Random Forest (PCA Öncesi)",
              "AlexNet (PCA Sonrası)", "Random Forest (PCA Sonrası)",
              "AlexNet (CARS Sonrası)", "Random Forest (CARS Sonrası)"],
    "Accuracy": [accuracy_orig, accuracy_rf_orig,
                 accuracy_pca, accuracy_rf_pca,
                 accuracy_cars, accuracy_rf_cars],
    "Precision": [precision_orig, precision_rf_orig,
                  precision_pca, precision_rf_pca,
                  precision_cars, precision_rf_cars],
    "Recall": [recall_orig, recall_rf_orig,
               recall_pca, recall_rf_pca,
               recall_cars, recall_rf_cars],
    "F1-Score": [f1_orig, f1_rf_orig,
                 f1_pca, f1_rf_pca,
                 f1_cars, f1_rf_cars]
}

results_df = pd.DataFrame(results)
print(results_df)

# Performans metriklerini görselleştirme
plot_performance_metrics(results_df)