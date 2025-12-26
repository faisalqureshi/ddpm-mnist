// DOM elements
const generateBtn = document.getElementById('generateBtn');
const refreshBtn = document.getElementById('refreshCheckpoints');
const imageContainer = document.getElementById('imageContainer');
const statusDiv = document.getElementById('status');
const outputDiv = document.getElementById('output');

const numImagesInput = document.getElementById('numImages');
const digitSelect = document.getElementById('digit');
const diffusionCkptInput = document.getElementById('diffusionCkpt');
const aeCkptInput = document.getElementById('aeCkpt');
const diffusionCkptSelect = document.getElementById('diffusionCkptSelect');
const aeCkptSelect = document.getElementById('aeCkptSelect');

// Load checkpoints on page load
loadCheckpoints();

// Event listeners
generateBtn.addEventListener('click', generateImages);
refreshBtn.addEventListener('click', loadCheckpoints);

// Sync dropdown with text input
diffusionCkptSelect.addEventListener('change', (e) => {
    diffusionCkptInput.value = e.target.value;
});

aeCkptSelect.addEventListener('change', (e) => {
    aeCkptInput.value = e.target.value;
});

// Load available checkpoints
async function loadCheckpoints() {
    try {
        showStatus('Loading checkpoints...', 'loading');

        const response = await fetch('/api/checkpoints');
        const data = await response.json();

        // Populate diffusion checkpoints
        diffusionCkptSelect.innerHTML = '<option value="">-- Select checkpoint --</option>';
        data.diffusion.forEach(ckpt => {
            const option = document.createElement('option');
            option.value = ckpt.path;
            option.textContent = ckpt.name;
            diffusionCkptSelect.appendChild(option);
        });

        // Populate autoencoder checkpoints
        aeCkptSelect.innerHTML = '<option value="">-- Select checkpoint --</option>';
        data.autoencoder.forEach(ckpt => {
            const option = document.createElement('option');
            option.value = ckpt.path;
            option.textContent = ckpt.name;
            aeCkptSelect.appendChild(option);
        });

        hideStatus();

        if (data.diffusion.length === 0 && data.autoencoder.length === 0) {
            showStatus('No checkpoints found. Please train models first or specify paths manually.', 'error');
        }
    } catch (error) {
        showStatus('Failed to load checkpoints: ' + error.message, 'error');
        console.error('Error loading checkpoints:', error);
    }
}

// Generate images
async function generateImages() {
    const numImages = parseInt(numImagesInput.value);
    const digit = digitSelect.value === '' ? null : parseInt(digitSelect.value);
    const diffusionCkpt = diffusionCkptInput.value.trim() || null;
    const aeCkpt = aeCkptInput.value.trim() || null;

    // Validation
    if (!diffusionCkpt) {
        showStatus('Please specify a diffusion checkpoint path', 'error');
        return;
    }

    if (numImages < 1 || numImages > 100) {
        showStatus('Number of images must be between 1 and 100', 'error');
        return;
    }

    // Disable button and show loading state
    generateBtn.disabled = true;
    generateBtn.classList.add('loading');

    // Clear previous image
    imageContainer.innerHTML = '<div class="placeholder"><p>Generating images...</p></div>';

    showStatus('Generating images... This may take a minute.', 'loading');
    outputDiv.classList.remove('show');
    outputDiv.textContent = '';

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                numImages,
                digit,
                diffusionCkpt,
                aeCkpt
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error + (data.details ? '\n\n' + data.details : ''));
        }

        // Display generated image
        const img = document.createElement('img');
        img.src = data.imageUrl + '?t=' + Date.now(); // Cache busting
        img.alt = 'Generated MNIST digits';
        img.onload = () => {
            imageContainer.innerHTML = '';
            imageContainer.appendChild(img);
        };
        img.onerror = () => {
            showStatus('Failed to load generated image', 'error');
        };

        // Show success message
        const digitMsg = digit !== null ? ` of digit ${digit}` : '';
        showStatus(`Successfully generated ${numImages} images${digitMsg}!`, 'success');

        // Show output log
        if (data.output) {
            outputDiv.textContent = data.output;
            outputDiv.classList.add('show');
        }

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        console.error('Generation error:', error);

        // Show error details in output
        outputDiv.textContent = error.message;
        outputDiv.classList.add('show');

        // Restore placeholder
        imageContainer.innerHTML = `
            <div class="placeholder">
                <svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="10" y="10" width="80" height="80" rx="4" stroke="#ccc" stroke-width="2" fill="none"/>
                    <path d="M30 50 L50 70 L70 30" stroke="#ccc" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <p>Generation failed</p>
            </div>
        `;
    } finally {
        // Re-enable button
        generateBtn.disabled = false;
        generateBtn.classList.remove('loading');
    }
}

// Status message helper
function showStatus(message, type = 'success') {
    statusDiv.textContent = message;
    statusDiv.className = 'status show ' + type;
}

function hideStatus() {
    statusDiv.classList.remove('show');
}

// Auto-hide success messages after 5 seconds
setInterval(() => {
    if (statusDiv.classList.contains('success')) {
        hideStatus();
    }
}, 5000);
