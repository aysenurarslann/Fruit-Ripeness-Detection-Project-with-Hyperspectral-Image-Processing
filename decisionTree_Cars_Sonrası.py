import os
import spectral.io.envi as envi
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import seaborn as sns

# Veriseti yükleme fonksiyonu
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

# Competitive Adaptive Reweighted Sampling (CARS) algoritması
def competitive_adaptive_reweighted_sampling(X, num_features):
    pca = PCA(n_components=num_features)
    pca.fit(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    selected_features = np.argsort(explained_variance_ratio)[::-1][:num_features]
    return selected_features

# Verisetini yükleme
main_folder_path = 'VIS_COR'  # Veri seti klasörü
img_datas, file_list = load_all_images(main_folder_path)
print(f"Toplam {len(img_datas)} görüntü yüklendi.")

# Örnekleme büyüklüğü
sample_size = 50


np.random.seed(0)  # Tekrarlanabilirlik için
sample_indices = np.random.choice(len(img_datas), sample_size, replace=False)
sampled_img_datas = [img_datas[i] for i in sample_indices]

# Tüm örneklemleri aynı boyutlara kırpma
min_height = min(img.shape[0] for img in sampled_img_datas)
min_width = min(img.shape[1] for img in sampled_img_datas)
cropped_img_datas = [img[:min_height, :min_width, :] for img in sampled_img_datas]

# Tüm örneklemleri birleştirerek yeniden şekillendirme
X_combined = np.vstack([img.reshape(-1, img.shape[-1]) for img in cropped_img_datas])

# PCA ile boyut azaltma
num_components_pca = 100  # PCA ile seçilecek bileşen sayısı
pca = PCA(n_components=num_components_pca)
X_pca = pca.fit_transform(X_combined)

# CARS ile bant seçimi
num_features_cars = 30  # Seçilecek bant sayısı CARS için
selected_bands_indices_cars = competitive_adaptive_reweighted_sampling(X_pca, num_features_cars)

# Seçilen bantları her bir görüntü için yeniden şekillendirme
selected_bands_cars = [img[:, :, selected_bands_indices_cars] for img in cropped_img_datas]

# Sınıf etiketleri
y = np.random.randint(0, 2, size=len(sampled_img_datas))

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(selected_bands_cars, y, test_size=0.2, random_state=0)

# Karar ağacı modeli oluşturma
clf = DecisionTreeClassifier(random_state=0)

# Modeli eğitme
X_train_flat = np.array([img.reshape(-1) for img in X_train])
X_test_flat = np.array([img.reshape(-1) for img in X_test])
clf.fit(X_train_flat, y_train)

# Modeli kullanarak test seti üzerinde tahmin yapma
y_pred = clf.predict(X_test_flat)
y_pred_proba = clf.predict_proba(X_test_flat)[:, 1]

# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Sonuçları yazdırma
print(f"Karar Ağacı ile eğitim sonuçları:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Karışıklık matrisi görselleştirme
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# ROC eğrisi görselleştirme
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (alan = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Karar Ağacı ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()

# İlk örneklemeden bir görüntü seçelim
sampled_image = cropped_img_datas[0]
selected_bands_image = selected_bands_cars[0]

# Görselleştirme için seçilen bantların sayısı
num_selected_bands = selected_bands_image.shape[-1]

# Her bir bant için görselleştirme
fig, axs = plt.subplots(1, num_selected_bands, figsize=(15, 5))
fig.suptitle('Seçilen Bantlar')

for i in range(num_selected_bands):
    axs[i].imshow(selected_bands_image[:, :, i], cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f' {selected_bands_indices_cars[i]}')

plt.tight_layout()
plt.show()
