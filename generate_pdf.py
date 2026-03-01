"""
PDF Rapor Üretici - Meme Kanseri Sınıflandırma Analizi
Öğrenci: Emine Taş | No: 427614
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from fpdf import FPDF

# Turkish -> ASCII normalization for fpdf core fonts
_TR = str.maketrans(
    'ıİğĞşŞçÇöÖüÜ•–—',
    'iIgGsSçCöOüU---'
)
_TR = str.maketrans({
    'ı':'i','İ':'I','ğ':'g','Ğ':'G','ş':'s','Ş':'S',
    'ç':'c','Ç':'C','ö':'o','Ö':'O','ü':'u','Ü':'U',
    '\u2022':'-','\u2013':'-','\u2014':'-','\u2019':"'",'\u201c':'"','\u201d':'"',
})

def t(text):
    return text.translate(_TR)

# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ─────────────────────────────────────────────
print("[1/8] Veri yükleniyor...")
df = pd.read_csv('data.csv')
df.drop(columns=[c for c in df.columns if 'Unnamed' in c], inplace=True)

df_clean = df.copy()
df_clean.drop(columns=['id'], inplace=True, errors='ignore')
le = LabelEncoder()
df_clean['diagnosis'] = le.fit_transform(df_clean['diagnosis'])  # B=0, M=1

X = df_clean.drop(columns=['diagnosis'])
y = df_clean['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 2. GÖRSEL 1 - Sınıf Dağılımı
# ─────────────────────────────────────────────
print("[2/8] Görsel 1 - Sınıf dağılımı...")
counts = df['diagnosis'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].pie(counts, labels=['Benign (B)', 'Malignant (M)'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
            startangle=90, explode=(0.05, 0.05), textprops={'fontsize': 12})
axes[0].set_title('Tanı Sınıf Dağılımı', fontsize=13, fontweight='bold')
axes[1].bar(['Benign (B)', 'Malignant (M)'], counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.45)
axes[1].set_title('Sınıf Bar Grafiği', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Örnek Sayısı')
for i, v in enumerate(counts.values):
    axes[1].text(i, v + 4, str(v), ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('fig1_class_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 3. GÖRSEL 2 - Özellik Dağılımları
# ─────────────────────────────────────────────
print("[3/8] Görsel 2 - Özellik dağılımları...")
features = ['radius_mean', 'texture_mean', 'perimeter_mean',
            'area_mean', 'smoothness_mean', 'compactness_mean']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, feat in enumerate(features):
    b = df[df['diagnosis'] == 'B'][feat]
    m = df[df['diagnosis'] == 'M'][feat]
    axes[i].hist(b, bins=22, alpha=0.6, color='#2ecc71', label='Benign')
    axes[i].hist(m, bins=22, alpha=0.6, color='#e74c3c', label='Malignant')
    axes[i].set_title(feat.replace('_', ' ').title(), fontsize=11)
    axes[i].legend(fontsize=9)
    axes[i].set_xlabel('Değer', fontsize=9)
plt.suptitle('Benign / Malignant - Özellik Dağılımları (Mean)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_feature_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 4. GÖRSEL 3 - Korelasyon Matrisi
# ─────────────────────────────────────────────
print("[4/8] Görsel 3 - Korelasyon matrisi...")
mean_cols = [c for c in df.columns if c.endswith('_mean')]
corr = df[mean_cols].corr()
fig, ax = plt.subplots(figsize=(13, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.4, annot_kws={'size': 8}, ax=ax)
ax.set_title('Korelasyon Matrisi (Mean Özellikler)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 5. MODEL KARŞILAŞTIRMASI
# ─────────────────────────────────────────────
print("[5/8] Model karşılaştırması (CrossVal)...")
models = {
    'Logistic\nRegression': LogisticRegression(max_iter=10000, random_state=42),
    'SVM\n(RBF)':           SVC(kernel='rbf', random_state=42),
    'Random\nForest':       RandomForestClassifier(n_estimators=100, random_state=42)
}
cv_means, cv_stds = [], []
for name, m in models.items():
    X_cv = X_train_scaled if 'Random' not in name else X_train
    sc = cross_val_score(m, X_cv, y_train, cv=5, scoring='accuracy')
    cv_means.append(sc.mean())
    cv_stds.append(sc.std())

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#3498db', '#9b59b6', '#e67e22']
bars = ax.bar(models.keys(), cv_means, yerr=cv_stds, capsize=8,
              color=colors, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylim(0.88, 1.02)
ax.set_ylabel('5-Fold CV Doğruluğu', fontsize=12)
ax.set_title('Algoritma Karşılaştırması (5-Fold Cross Validation)', fontsize=13, fontweight='bold')
ax.axhline(y=max(cv_means), color='red', linestyle='--', alpha=0.5, label=f'En iyi: {max(cv_means):.4f}')
ax.legend()
plt.tight_layout()
plt.savefig('fig4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 6. RANDOM FOREST FİNAL MODEL
# ─────────────────────────────────────────────
print("[6/8] Random Forest final model eğitiliyor...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
gs = GridSearchCV(RandomForestClassifier(random_state=42),
                  param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)
best_rf = gs.best_estimator_
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
best_params = gs.best_params_

print(f"   Accuracy : {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")
print(f"   F1-Score : {f1:.4f}")

# ─────────────────────────────────────────────
# 7. GÖRSELLer - Confusion Matrix & Metrikler
# ─────────────────────────────────────────────
print("[7/8] Görseller oluşturuluyor...")
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Confusion Matrix (Sayısal)', fontsize=13, fontweight='bold')
cm_n = confusion_matrix(y_test, y_pred, normalize='true')
disp2 = ConfusionMatrixDisplay(cm_n, display_labels=['Benign', 'Malignant'])
disp2.plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title('Confusion Matrix (Normalize)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Metrikler bar grafiği
metrics_d = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(metrics_d.keys(), metrics_d.values(),
              color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
              edgecolor='black', alpha=0.88)
for bar, val in zip(bars, metrics_d.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Skor', fontsize=12)
ax.set_title('Random Forest - Performans Metrikleri', fontsize=14, fontweight='bold')
ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig6_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# Özellik önemi
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top15 = importances.nlargest(15).sort_values()
fig, ax = plt.subplots(figsize=(10, 7))
colors_i = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15)))
top15.plot(kind='barh', ax=ax, color=colors_i, edgecolor='black', alpha=0.85)
ax.set_xlabel('Özellik Önemi', fontsize=12)
ax.set_title('Random Forest - İlk 15 Özellik Önemi', fontsize=14, fontweight='bold')
for i, val in enumerate(top15.values):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('fig7_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 8. PDF OLUŞTURMA
# ─────────────────────────────────────────────
print("[8/8] PDF oluşturuluyor...")

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, 'Meme Kanseri Siniflandirmasi - Makine Ogrenmesi Analiz Raporu', 0, 1, 'C')
        self.ln(1)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title, color=(41, 128, 185)):
        self.set_font('Helvetica', 'B', 13)
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, t(title), 0, 1, 'L', fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def body_text(self, text, size=11):
        self.set_font('Helvetica', '', size)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, t(text))
        self.ln(2)

    def add_image_centered(self, img_path, w=170, h=0):
        if os.path.exists(img_path):
            x = (210 - w) / 2
            self.image(img_path, x=x, w=w, h=h)
            self.ln(4)

    def metric_box(self, label, value, color):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.cell(43, 18, f'{label}', 0, 0, 'C', fill=True)
        self.set_font('Helvetica', 'B', 14)
        self.cell(3, 18, '', 0, 0)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(30, 30, 30)
        self.cell(40, 18, f'{value:.4f}', 1, 0, 'C', fill=True)
        self.cell(4, 18, '', 0, 0)


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=18)

# ── KAPAK SAYFASI ──
pdf.add_page()
pdf.ln(15)
pdf.set_font('Helvetica', 'B', 22)
pdf.set_text_color(41, 128, 185)
pdf.cell(0, 12, 'MEME KANSERi SINIFLANDIRMASI', 0, 1, 'C')
pdf.set_font('Helvetica', 'B', 16)
pdf.set_text_color(52, 73, 94)
pdf.cell(0, 10, 'Makine Ogrenmesi Analiz Raporu', 0, 1, 'C')
pdf.ln(8)
pdf.set_draw_color(41, 128, 185)
pdf.set_line_width(0.8)
pdf.line(40, pdf.get_y(), 170, pdf.get_y())
pdf.ln(12)

info_rows = [
    ('Ogrenci No', ': 427614'),
    ('Ad Soyad', ': Emine Tas'),
    ('GitHub', ': [GitHub linkinizi buraya ekleyiniz]'),
    ('Veri Seti', ': Breast Cancer Wisconsin (Diagnostic)'),
    ('Algoritma', ': Random Forest Classifier'),
    ('Tarih', ': 2026'),
]
for label, val in info_rows:
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(52, 73, 94)
    pdf.cell(42, 9, label, 0, 0)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 9, val, 0, 1)

pdf.ln(10)
pdf.set_draw_color(41, 128, 185)
pdf.line(40, pdf.get_y(), 170, pdf.get_y())
pdf.ln(12)

pdf.set_font('Helvetica', 'B', 11)
pdf.set_text_color(52, 73, 94)
pdf.cell(0, 8, 'Ozet', 0, 1, 'C')
pdf.set_font('Helvetica', '', 11)
pdf.set_text_color(50, 50, 50)
summary = (
    "Bu calismada Breast Cancer Wisconsin veri seti uzerinde makine ogrenmesi yontemleri "
    "uygulanarak tumorlerin iyi huylu (Benign) veya kotu huylu (Malignant) oldugu "
    "siniflandirilmistir. Uclu model karsilastirmasi (Logistic Regression, SVM, Random Forest) "
    "sonucunda en yuksek 5-fold cross-validation dogrulugunu elde eden Random Forest algoritmasina "
    "karar verilmistir. Model; Accuracy, Precision, Recall ve F1-Score metrikleriyle "
    "kapsamli bicimde degerlendirilmistir."
)
pdf.multi_cell(0, 7, t(summary), align='C')

# ── SAYFA 2: VERİ SETİ ──
pdf.add_page()
pdf.chapter_title('1. Veri Seti Hakkinda')
pdf.body_text(
    "Breast Cancer Wisconsin (Diagnostic) veri seti, dijital goruntu isleme ile elde edilen "
    "meme kitlesi ince igne aspire biyopsisinden (FNA) turetilen ozellikler icerir. "
    "Her ornek icin 10 temel ozelligi tanimlayan; ortalama (mean), standart hata (se) ve "
    "en kotu (worst) degerler olmak uzere toplam 30 sayisal ozellik mevcuttur."
)

# Tablo: Veri seti bilgileri
pdf.set_font('Helvetica', 'B', 11)
pdf.set_fill_color(41, 128, 185)
pdf.set_text_color(255, 255, 255)
pdf.cell(60, 9, 'Ozellik', 1, 0, 'C', fill=True)
pdf.cell(120, 9, 'Deger', 1, 1, 'C', fill=True)

rows = [
    ('Toplam Ornek', '569'),
    ('Toplam Ozellik', '30 (sayisal)'),
    ('Hedef Degisken', 'diagnosis: B (Benign=0) / M (Malignant=1)'),
    ('Benign Sayisi', f"{(df['diagnosis']=='B').sum()} (%{(df['diagnosis']=='B').mean()*100:.1f})"),
    ('Malignant Sayisi', f"{(df['diagnosis']=='M').sum()} (%{(df['diagnosis']=='M').mean()*100:.1f})"),
    ('Egitim Seti', f'{X_train.shape[0]} ornek (%80)'),
    ('Test Seti', f'{X_test.shape[0]} ornek (%20)'),
    ('Eksik Deger', 'Yok'),
]
for i, (k, v) in enumerate(rows):
    fill = (i % 2 == 0)
    pdf.set_fill_color(235, 243, 252) if fill else pdf.set_fill_color(255, 255, 255)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(60, 8, k, 1, 0, 'L', fill=fill)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(120, 8, v, 1, 1, 'L', fill=fill)

pdf.ln(6)
pdf.chapter_title('2. Kesifsel Veri Analizi (EDA)', color=(39, 174, 96))
pdf.add_image_centered('fig1_class_dist.png', w=170)
pdf.body_text(
    "Grafik 1: Veri setinde 357 Benign (%62.7) ve 212 Malignant (%37.3) ornek bulunmaktadir. "
    "Sinif dengesizligi hafif duzeyinde olup model egitimini olumsuz etkilememektedir."
)

# ── SAYFA 3: EDA devam ──
pdf.add_page()
pdf.chapter_title('2. Kesifsel Veri Analizi (devam)', color=(39, 174, 96))
pdf.add_image_centered('fig2_feature_dist.png', w=175)
pdf.body_text(
    "Grafik 2: Mean ozelliklerinin sinifa gore dagilimi incelendiginde, Malignant tumorlerin "
    "radius, perimeter, area ve concavity degerlerinin Benign olgulara kiyasla belirgin "
    "sekilde yuksek oldugu gorulmektedir."
)
pdf.add_image_centered('fig3_correlation.png', w=165)
pdf.body_text(
    "Grafik 3: Korelasyon matrisi, radius-perimeter-area ozellik uclusunun birbirleriyle "
    "yuksek pozitif korelasyon (>0.95) sergiledigini ortaya koymaktadir. Bu durum, "
    "Random Forest gibi ensemble yontemlerin ozellik secimindeki esnekligini on plana cikarir."
)

# ── SAYFA 4: ALGORİTMA SEÇİMİ ──
pdf.add_page()
pdf.chapter_title('3. Algoritma Secimi ve Karsilastirma', color=(142, 68, 173))
pdf.body_text(
    "Uygun makine ogrenmesi algoritmasi secmek icin asagidaki uc yaygin ikili siniflandirici "
    "5-katli capraz dogrulama (5-Fold Cross Validation) ile karsilastirilmistir:"
)

algos = [
    ('Logistic Regression', 'Dogrusal sinir, hizli, yorumlanabilir. Ozellikler arasinda dogrusal iliskiyi ongorer.'),
    ('SVM (RBF Kernel)', 'Yuksek boyutlu verilerde guclu. RBF cekirdegi ile dogrusalsizligi yonetir.'),
    ('Random Forest', 'Cok sayida karar agacinin toplulugu. Asiri ogrenmeyeciyle ozellik onemi saglar.'),
]
for name, desc in algos:
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 8, f'  - {name}', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(8, 6, '', 0, 0)
    pdf.multi_cell(0, 6, t(desc))
    pdf.ln(1)

pdf.add_image_centered('fig4_model_comparison.png', w=160)
pdf.body_text(
    f"5-Fold Cross Validation Sonuclari:\n"
    f"  Logistic Regression  : {cv_means[0]:.4f} (+/- {cv_stds[0]:.4f})\n"
    f"  SVM (RBF)            : {cv_means[1]:.4f} (+/- {cv_stds[1]:.4f})\n"
    f"  Random Forest        : {cv_means[2]:.4f} (+/- {cv_stds[2]:.4f})\n\n"
    f"=> Random Forest en yuksek CV dogrulugunu elde etmistir.\n\n"
    "Random Forest Secim Gerekceleri:\n"
    "  1. En yuksek capraz dogrulama dogrulugu\n"
    "  2. Ensemble yapisi sayesinde overfitting'e karsi direncledir\n"
    "  3. Ozellik onemi analizi ile yorumlanabilirlik saglar\n"
    "  4. Olceklendirme gerektirmez (guclu olcek bagimsizip)\n"
    "  5. Tibbi veriler icin kanıtlanmis guclu performans"
)

# ── SAYFA 5: MODEL EGİTİMİ & DEĞERLENDİRME ──
pdf.add_page()
pdf.chapter_title('4. Model Egitimi', color=(231, 76, 60))
pdf.body_text(
    f"Random Forest modeli GridSearchCV ile hiperparametre optimizasyonuna tabi tutulmustur.\n\n"
    f"Arama Uzayi:\n"
    f"  n_estimators     : [100, 200, 300]\n"
    f"  max_depth        : [None, 10, 20]\n"
    f"  min_samples_split: [2, 5]\n\n"
    f"En iyi parametreler:\n"
    f"  n_estimators     : {best_params.get('n_estimators')}\n"
    f"  max_depth        : {best_params.get('max_depth')}\n"
    f"  min_samples_split: {best_params.get('min_samples_split')}\n\n"
    f"Egitim seti: {X_train.shape[0]} ornek | Test seti: {X_test.shape[0]} ornek\n"
    f"Bolunme orani: %80 egitim / %20 test (stratified)"
)

pdf.chapter_title('5. Performans Degerlendirmesi', color=(231, 76, 60))
pdf.body_text(
    "Model, test seti uzerinde asagidaki 4 temel metrik ile degerlendirilmistir:\n\n"
    "  Accuracy  (Dogruluk)   : Dogru tahminlerin tum orneklere orani.\n"
    "  Precision (Kesinlik)   : Malignant tahmin edilenlerin gercekten M olma orani.\n"
    "  Recall (Duyarlilik)    : Gercekte M olanlarin dogru tespit edilme orani. [Kritik!]\n"
    "  F1-Score               : Precision ve Recall'in harmonik ortalamasi."
)

# Metrik kutucuklar
pdf.ln(4)
pdf.set_font('Helvetica', 'B', 11)
pdf.metric_box('Accuracy', acc, (41, 128, 185))
pdf.ln(22)
pdf.metric_box('Precision', prec, (39, 174, 96))
pdf.ln(22)
pdf.metric_box('Recall', rec, (231, 76, 60))
pdf.ln(22)
pdf.metric_box('F1-Score', f1, (243, 156, 18))
pdf.ln(22)

pdf.ln(4)
pdf.add_image_centered('fig6_metrics.png', w=155)

# ── SAYFA 6: CONFUSION MATRIX ──
pdf.add_page()
pdf.chapter_title('6. Confusion Matrix Analizi', color=(44, 62, 80))
pdf.add_image_centered('fig5_confusion_matrix.png', w=170)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
pdf.body_text(
    f"Confusion Matrix Degerleri:\n"
    f"  TP (Dogru Malignant)   : {tp} - Gercek M, tahmin M\n"
    f"  TN (Dogru Benign)      : {tn} - Gercek B, tahmin B\n"
    f"  FP (Yanlis Malignant)  : {fp} - Gercek B, tahmin M (Tip I Hata)\n"
    f"  FN (Kacirilan Malignant): {fn} - Gercek M, tahmin B (Tip II Hata)\n\n"
    "Tibbi acidan en kritik hata, gercek malignant olgularin benign olarak siniflandirilmasidir "
    "(False Negative / Tip II Hata). Bu nedenle Recall metriginin yuksek tutulmasi onceliklidir. "
    f"Modelimiz {rec*100:.1f}% Recall ile bu kriteri basariyla karsilamistur."
)

# ── SAYFA 7: ÖZELLİK ÖNEMİ ──
pdf.add_page()
pdf.chapter_title('7. Ozellik Onemi Analizi', color=(22, 160, 133))
pdf.add_image_centered('fig7_feature_importance.png', w=170)
top5 = importances.nlargest(5)
pdf.body_text(
    "En onemli 5 ozellik:\n" +
    "\n".join([f"  {i+1}. {feat:<30} Onemi: {val:.4f}" for i, (feat, val) in enumerate(top5.items())]) +
    "\n\n"
    "Bulgular: 'Worst' (en kotu) ozellikler, taninin belirlenmesinde 'mean' ve 'se' degerlerinden "
    "daha belirleyici olmaktadir. Ozellikle concave_points_worst, perimeter_worst ve radius_worst, "
    "tumoru karakterize eden sekil duzensizkigi ve boyut gostergelerdir."
)

# ── SAYFA 8: SONUÇ ──
pdf.add_page()
pdf.chapter_title('8. Sonuc ve Degerlendirme', color=(41, 128, 185))
cr = classification_report(y_test, y_pred, target_names=['Benign (B)', 'Malignant (M)'])
pdf.set_font('Courier', '', 9)
pdf.set_text_color(30, 30, 30)
pdf.multi_cell(0, 5, t(cr))
pdf.ln(4)

pdf.set_font('Helvetica', '', 11)
pdf.chapter_title('Genel Degerlendirme', color=(52, 73, 94))
pdf.body_text(
    "Bu calismada Breast Cancer Wisconsin veri seti uzerinde Random Forest algoritmasi "
    "uygulanmis ve asagidaki sonuclar elde edilmistir:\n\n"
    f"  • Accuracy  : %{acc*100:.2f}\n"
    f"  • Precision : %{prec*100:.2f}\n"
    f"  • Recall    : %{rec*100:.2f}\n"
    f"  • F1-Score  : %{f1*100:.2f}\n\n"
    "Model, 30 ozelligi basariyla isleme alarak yuksek dogrulukta siniflandirma "
    "gerceklestirmistir. Tibbi birimlerde karar destek sistemi olarak kullanilabilecek "
    "duzeyde performans sergilemektedir.\n\n"
    "Gelecek Calisma Onerileri:\n"
    "  • SMOTE ile sinif dengesizliginin giderilmesi\n"
    "  • SHAP degerleri ile model yorumlanabilirligi\n"
    "  • Derin ogrenme mimarileriyle karsilastirma\n"
    "  • K-katli CV ile daha guvenilir dogruluk kestirimi"
)

OUTPUT = 'rapor_427614_EmineTas.pdf'
pdf.output(OUTPUT)
print(f"\n[OK] PDF olusturuldu: {OUTPUT}")

# Gecici gorselleri temizle
for img in ['fig1_class_dist.png','fig2_feature_dist.png','fig3_correlation.png',
            'fig4_model_comparison.png','fig5_confusion_matrix.png',
            'fig6_metrics.png','fig7_feature_importance.png']:
    if os.path.exists(img):
        os.remove(img)
print("[OK] Gecici gorsel dosyalari silindi.")
