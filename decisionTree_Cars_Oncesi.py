import os
import spectral.io.envi as envi
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
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

# Verisetini yükleme
main_folder_path = 'VIS_COR'  # Veri seti klasörü
img_datas, file_list = load_all_images(main_folder_path)


# Örnekleme büyüklüğü
sample_size = 50  # Örneklem büyüklüğünü burada ayarlayabilirsiniz


np.random.seed(0)  # Tekrarlanabilirlik için
sample_indices = np.random.choice(len(img_datas), sample_size, replace=False)
sampled_img_datas = [img_datas[i] for i in sample_indices]

# Tüm örneklemleri aynı boyutlara kırpma
min_height = min(img.shape[0] for img in sampled_img_datas)
min_width = min(img.shape[1] for img in sampled_img_datas)
cropped_img_datas = [img[:min_height, :min_width, :] for img in sampled_img_datas]

# Sınıf etiketleri
y = np.random.randint(0, 2, size=sample_size)  # sample_size kadar rastgele sınıf etiketi oluşturuldu

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(cropped_img_datas, y, test_size=0.2, random_state=0)

# Karar ağacı modeli oluşturma
clf = DecisionTreeClassifier(random_state=0)

# Modeli eğitme
X_train_flat = np.array([img.reshape(-1) for img in X_train])
X_test_flat = np.array([img.reshape(-1) for img in X_test])
clf.fit(X_train_flat, y_train)

# Modeli kullanarak test seti üzerinde tahmin yapma
y_pred = clf.predict(X_test_flat)

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

# Konfüzyon matrisi
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC eğrisi ve AUC
y_pred_prob = clf.predict_proba(X_test_flat)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

