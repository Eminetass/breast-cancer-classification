# Breast Cancer Wisconsin - Makine Öğrenmesi ile Sınıflandırma

**Öğrenci No:** 427614
**Ad Soyad:** Emine Taş

---

## Proje Hakkında

Bu proje, **Breast Cancer Wisconsin** veri seti kullanılarak meme tümörlerinin **iyi huylu (Benign)** veya **kötü huylu (Malignant)** olarak sınıflandırılmasını amaçlamaktadır. Sınıflandırma için **Random Forest** algoritması seçilmiş ve Accuracy, Precision, Recall, F1-Score metrikleriyle değerlendirilmiştir.

---

## Veri Seti

| Özellik | Bilgi |
|---------|-------|
| Kaynak | Breast Cancer Wisconsin (Diagnostic) |
| Örnek Sayısı | 569 |
| Özellik Sayısı | 30 (sayısal) |
| Hedef Değişken | diagnosis: B (Benign=0) / M (Malignant=1) |
| Sınıf Dağılımı | %62.7 Benign, %37.3 Malignant |

**Özellikler:** Tümörün radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry ve fractal dimension değerlerinin **mean**, **standard error** ve **worst** istatistikleri.

---

## Kullanılan Teknolojiler

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- fpdf2

---

## Proje Yapısı

```
breast-cancer-classification/
├── breast_cancer_classification.ipynb   # Ana analiz notebook'u
├── data.csv                             # Veri seti
├── generate_pdf.py                      # PDF rapor üreticisi
├── README.md                            # Bu dosya
└── rapor_427614_EmineTas.pdf           # Analiz raporu (oluşturulan)
```

---

## Algoritma Seçimi: Neden Random Forest?

Üç algoritma 5-katlı çapraz doğrulama ile karşılaştırılmıştır:

| Algoritma | CV Accuracy |
|-----------|-------------|
| Logistic Regression | ~%95.4 |
| SVM (RBF Kernel) | ~%97.1 |
| **Random Forest** | **~%97.4** |

**Random Forest** şu nedenlerle tercih edilmiştir:
1. En yüksek cross-validation doğruluğu
2. Overfitting'e karşı dirençli (ensemble yapısı)
3. Özellik önemi analizi imkanı
4. Ölçeklendirme gerektirmez
5. Tıbbi veriler için sağlam performans

---

## Performans Metrikleri

| Metrik | Değer |
|--------|-------|
| **Accuracy** | ~97% |
| **Precision** | ~97% |
| **Recall** | ~95% |
| **F1-Score** | ~96% |

> **Not:** Recall (Duyarlılık) tıbbi tanıda kritik öneme sahiptir. Gerçek malignant vakaların gözden kaçırılması (False Negative) klinik açıdan son derece risklidir.

---

## Kurulum ve Çalıştırma

```bash
# Gerekli kütüphaneleri kur
pip install pandas numpy matplotlib seaborn scikit-learn fpdf2

# Jupyter Notebook'u başlat
jupyter notebook breast_cancer_classification.ipynb

# PDF raporu oluştur
python generate_pdf.py
```

---

## Önemli Bulgular

- En belirleyici özellikler: `concave points_worst`, `perimeter_worst`, `radius_worst`, `area_worst`
- Tümör boyutu ve şekil düzensizliği malignant tanısı için en güçlü belirteçlerdir
- Model, %80/%20 eğitim-test bölünmesi ile değerlendirilmiştir
- GridSearchCV ile hiperparametre optimizasyonu yapılmıştır

---

## Lisans

Bu proje akademik ödev amacıyla hazırlanmıştır.
