// Wait until the DOM is fully loaded before running the script
document.addEventListener('DOMContentLoaded', () => {
    // API endpoints for CCT and ViT models
    const CCT_API_URL = "http://127.0.0.1:5000/cct";
    const VIT_API_URL = "http://127.0.0.1:5000/vit";

    // DOM elements for file input, image preview, loading spinner, and buttons
    const fileInput = document.getElementById('fileInput');           // File input element
    const imagePreview = document.getElementById('imagePreview');     // Image preview element
    const loadingElement = document.querySelector('.loading');         // Loading spinner element
    const submitCCTBtn = document.getElementById('submitCCT');        // Button to submit image for CCT analysis
    const submitVITBtn = document.getElementById('submitVIT');        // Button to submit image for ViT analysis

    let selectedFile = null; // Variable to store the selected file

    /**
     * Event listener for file input change.
     * Updates the `selectedFile` and displays the image preview.
     */
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            previewImage(selectedFile);
        }
    });

    /**
     * Event listener for the CCT submit button click.
     * Sends the selected file to the CCT API for analysis.
     */
    submitCCTBtn.addEventListener('click', () => {
        if (selectedFile) {
            analyzeImage(selectedFile, CCT_API_URL, 'cctContent');
        } else {
            alert('Please select an image first!');
        }
    });

    /**
     * Event listener for the ViT submit button click.
     * Sends the selected file to the ViT API for analysis.
     */
    submitVITBtn.addEventListener('click', () => {
        if (selectedFile) {
            analyzeImage(selectedFile, VIT_API_URL, 'vitContent');
        } else {
            alert('Please select an image first!');
        }
    });

    /**
     * Displays a preview of the selected image.
     * @param {File} file - The selected image file.
     */
    function previewImage(file) {
        // Check if the file is a valid image
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            return;
        }

        // Create a FileReader to read the image file
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;     // Set the preview image source
            imagePreview.style.display = 'block';   // Display the image preview
        };
        reader.readAsDataURL(file); // Read the file as a Data URL
    }

    /**
     * Sends the selected image to the specified API for analysis.
     * @param {File} file - The selected image file.
     * @param {string} apiUrl - The API endpoint to send the image to.
     * @param {string} resultContainerId - The ID of the element to display results in.
     */
    async function analyzeImage(file, apiUrl, resultContainerId) {
        loadingElement.style.display = 'block'; // Show the loading spinner
        const resultContainer = document.getElementById(resultContainerId);
        resultContainer.innerHTML = ''; // Clear previous results

        // Create a FormData object to send the image file in the request
        const formData = new FormData();
        formData.append('image', file);

        try {
            // Send a POST request to the specified API with the image file
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData
            });

            // Parse the JSON response
            const results = await response.json();
            resultContainer.innerHTML = createResultHTML(results); // Display the results
        } catch (error) {
            // Handle errors by logging to the console and displaying an error message
            console.error(`Error during analysis with ${apiUrl}:`, error);
            resultContainer.innerHTML = '<p class="error">Error during analysis. Please try again.</p>';
        }

        loadingElement.style.display = 'none'; // Hide the loading spinner after the request completes
    }

    /**
     * Generates HTML to display the analysis results.
     * @param {Object} results - The analysis results from the API.
     * @returns {string} - HTML string to display the results.
     */
    function createResultHTML(results) {
        // Check if the results contain an error message
        if (results.error) {
            return `<p class="error"><strong>Error:</strong> ${results.error}</p>`;
        }

        // Parse confidence score and create HTML to display the predicted class and confidence
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
