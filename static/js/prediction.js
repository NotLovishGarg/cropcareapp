// js/prediction.js
document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const predictBtn = document.getElementById('predict-btn');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');
    const removeImageBtn = document.getElementById('remove-image');

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight);
    });

    function highlight() {
        dropZone.classList.add('dragover');
    }

    function unhighlight() {
        dropZone.classList.remove('dragover');
    }

    // Handle file drop
    dropZone.addEventListener('drop', handleDrop);
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            handleFile(files[0]);
        }
    }

    // Handle file selection via button
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle file preview and validation
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            document.querySelector('.drop-zone-content').classList.add('hidden');
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Handle image removal
    removeImageBtn.addEventListener('click', () => {
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        document.querySelector('.drop-zone-content').classList.remove('hidden');
        predictBtn.disabled = true;
        fileInput.value = '';
    });

    // Handle prediction
    predictBtn.addEventListener('click', async () => {
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        try {
            predictBtn.disabled = true;
            predictBtn.textContent = 'Predicting...';

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (result.disease) {
                window.location.href = `/${result.disease}`;
                // alert(`Predicted Disease: ${result.disease}`);
            } else {
                alert('Error predicting disease. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error predicting disease. Please try again.');
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict Disease';
        }
    });
});