# QARI v0.3 Integration Guide

دليل تكامل QARI v0.3 (Quality Arabic Recognition Interface) مع التطبيق.

## ما هو QARI v0.3؟

QARI هو نظام OCR متقدم مخصص للعربية يوفر:
- دقة عالية في التعرف على النص العربي
- دعم كامل للعربية والإنجليزية
- الحفاظ على تخطيط النص
- معالجة متقدمة للصور

## التثبيت

### الطريقة 1: إذا كان QARI متاحاً كحزمة Python

```bash
pip install qari
# أو
pip install qari-ocr
```

### الطريقة 2: التثبيت من المصدر

إذا كان QARI متاحاً على GitHub:

```bash
git clone https://github.com/[qari-repo]/qari.git
cd qari
pip install -e .
```

### الطريقة 3: Fallback تلقائي

إذا لم يكن QARI متاحاً، سيستخدم الكود **PaddleOCR** تلقائياً كبديل.

## الاستخدام

### في التطبيق Flask

التطبيق يستخدم QARI v0.3 تلقائياً. لا حاجة لتغيير أي شيء!

```python
# app.py يستخدم QARI تلقائياً
python app.py
```

### استخدام مباشر

```python
from qari_ocr import QARIOCR, process_image_text

# استخدام بسيط
text = process_image_text("image.jpg")

# مع تصحيح LLM
text = process_image_text("image.jpg", llm_model="mistral")

# استخدام الكلاس مباشرة
processor = QARIOCR(use_gpu=False, enable_llm_postprocessing=True)
text = processor.process_image_text("image.jpg")
```

## المميزات

### 1. Preprocessing تلقائي
- تحويل إلى Grayscale
- تحسين التباين (CLAHE)
- إزالة الضوضاء

### 2. معالجة النص العربي
- ربط الأحرف باستخدام `arabic_reshaper`
- تصحيح اتجاه RTL باستخدام `bidi.algorithm`

### 3. الحفاظ على التخطيط
- الحفاظ على الأسطر
- الحفاظ على المسافات

### 4. تصحيح اختياري عبر LLM
- تصحيح الأخطاء البسيطة
- الحفاظ على التخطيط

## التكوين

### تفعيل GPU

```python
processor = QARIOCR(use_gpu=True)
```

### تفعيل LLM Postprocessing

```python
processor = QARIOCR(
    enable_llm_postprocessing=True,
    llm_model="mistral"
)
```

## استكشاف الأخطاء

### QARI غير متاح

إذا ظهرت رسالة:
```
Using PaddleOCR as fallback (QARI not found)
```

هذا طبيعي! الكود سيستخدم PaddleOCR تلقائياً.

### تثبيت QARI يدوياً

إذا كان لديك QARI v0.3 محلياً:

1. ضع ملفات QARI في مجلد `qari/`
2. أو ثبتها كحزمة Python محلية

### تحديث API

إذا تغيرت واجهة برمجة QARI، يمكن تحديث `qari_ocr.py`:

```python
# في extract_text_qari()
# جرب طرق مختلفة:
result = self.qari.recognize(image_path)  # أو
result = self.qari.ocr(image_path)        # أو
result = self.qari.extract_text(image_path)
```

## ملاحظات

- الكود يدعم QARI v0.3 مع fallback تلقائي لـ PaddleOCR
- لا حاجة لاتصال بالإنترنت بعد التثبيت
- كل شيء يعمل محلياً

## الدعم

إذا كان لديك معلومات عن QARI v0.3 أو كيفية تثبيته، يرجى تحديث هذا الملف!

