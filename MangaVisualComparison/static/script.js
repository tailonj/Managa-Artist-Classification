document.addEventListener('DOMContentLoaded', () => {
    const CCT_API_URL = "http://127.0.0.1:5000/cct";
    const VIT_API_URL = "http://127.0.0.1:5000/vit";

    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const loadingElement = document.querySelector('.loading');
    const submitCCTBtn = document.getElementById('submitCCT');
    const submitVITBtn = document.getElementById('submitVIT');

    let selectedFile = null; // Store the selected file

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            previewImage(selectedFile);
        }
    });

    submitCCTBtn.addEventListener('click', () => {
        if (selectedFile) {
            analyzeImage(selectedFile, CCT_API_URL, 'cctContent');
        } else {
            alert('Please select an image first!');
        }
    });

    submitVITBtn.addEventListener('click', () => {
        if (selectedFile) {
            analyzeImage(selectedFile, VIT_API_URL, 'vitContent');
        } else {
            alert('Please select an image first!');
        }
    });

    function previewImage(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    async function analyzeImage(file, apiUrl, resultContainerId) {
        loadingElement.style.display = 'block';
        const resultContainer = document.getElementById(resultContainerId);
        resultContainer.innerHTML = ''; // Clear previous results

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData
            });

            const results = await response.json();
            resultContainer.innerHTML = createResultHTML(results);
        } catch (error) {
            console.error(`Error during analysis with ${apiUrl}:`, error);
            resultContainer.innerHTML = '<p class="error">Error during analysis. Please try again.</p>';
        }

        loadingElement.style.display = 'none'; // Hide loading spinner
    }

    function createResultHTML(results) {
        if (results.error) {
            return `<p class="error"><strong>Error:</strong> ${results.error}</p>`;
        }
        const confidence = parseFloat(results.confidence);
        return `
            <p><strong>Predicted Artist:</strong> ${results.predicted_class}</p>
            <p><strong>Confidence:</strong> ${results.confidence}</p>
            <div class="confidence-bar">
                <div class="confidence-progress" style="width: ${confidence}%"></div>
            </div>
        `;
    }
});
