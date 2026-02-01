document.getElementById('ocrForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    const form = e.target;
    const fileInput = document.getElementById('imageInput');
    const ocrBtn = document.getElementById('ocrBtn');
    const ocrTextarea = document.getElementById('ocrText');
    const llmTextarea = document.getElementById('llmText');
    
    if (!fileInput.files.length) {
        alert('يرجى اختيار صورة');
        return;
    }

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    // Update UI to show processing state
    ocrBtn.textContent = "جارٍ التحويل ...";
    ocrBtn.disabled = true;
    ocrTextarea.value = '';
    llmTextarea.value = '';

    try {
        const response = await fetch('/ocr', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            alert('خطأ: ' + result.error);
        } else {
            ocrTextarea.value = result.ocr_text || '';
            llmTextarea.value = result.llm_text || '';
        }
    } catch (err) {
        alert("حدث خطأ أثناء المعالجة: " + err.message);
        console.error('Error:', err);
    } finally {
        // Reset button state
        ocrBtn.textContent = "ابدأ التحويل";
        ocrBtn.disabled = false;
    }
});

