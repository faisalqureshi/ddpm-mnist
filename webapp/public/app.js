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

// Process visualization elements
const visualizeProcessCheckbox = document.getElementById('visualizeProcess');
const processStepsGroup = document.getElementById('processStepsGroup');
const processStepsInput = document.getElementById('processSteps');
const animationControls = document.getElementById('animationControls');
const playPauseBtn = document.getElementById('playPauseBtn');
const restartBtn = document.getElementById('restartBtn');
const frameSlider = document.getElementById('frameSlider');
const frameCounter = document.getElementById('frameCounter');
const speedControl = document.getElementById('speedControl');
const speedLabel = document.getElementById('speedLabel');

// Animation state
let animationFrames = [];
let currentFrame = 0;
let isPlaying = false;
let animationInterval = null;

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

// Process visualization checkbox
visualizeProcessCheckbox.addEventListener('change', (e) => {
    processStepsGroup.style.display = e.target.checked ? 'block' : 'none';
    if (!e.target.checked) {
        stopAnimation();
        animationControls.style.display = 'none';
    }
});

// Animation control event listeners
playPauseBtn.addEventListener('click', togglePlayPause);
restartBtn.addEventListener('click', restartAnimation);
frameSlider.addEventListener('input', (e) => {
    currentFrame = parseInt(e.target.value);
    updateFrame();
    if (isPlaying) {
        stopAnimation();
    }
});
speedControl.addEventListener('input', (e) => {
    const fps = parseInt(e.target.value);
    speedLabel.textContent = `${fps} FPS`;
    if (isPlaying) {
        stopAnimation();
        startAnimation();
    }
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
    const visualizeProcess = visualizeProcessCheckbox.checked;
    const processSteps = parseInt(processStepsInput.value);

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

    // Clear previous image and stop any existing animation
    stopAnimation();
    animationControls.style.display = 'none';
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
                aeCkpt,
                visualizeProcess,
                processSteps
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error + (data.details ? '\n\n' + data.details : ''));
        }

        // Handle process visualization
        if (data.type === 'process' && data.frames) {
            await setupProcessVisualization(data.frames);
            const digitMsg = digit !== null ? ` of digit ${digit}` : '';
            showStatus(`Successfully generated ${numImages} images${digitMsg} with ${data.numSteps} denoising steps!`, 'success');
        } else {
            // Display standard generated image
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
        }

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

// ===== FILE BROWSER =====

const fileBrowserModal = document.getElementById('fileBrowserModal');
const modalClose = document.querySelector('.modal-close');
const fileList = document.getElementById('fileList');
const breadcrumb = document.getElementById('breadcrumb');
const selectedPathSpan = document.getElementById('selectedPath');
const selectFileBtn = document.getElementById('selectFileBtn');
const cancelBrowseBtn = document.getElementById('cancelBrowseBtn');
const parentDirBtn = document.getElementById('parentDirBtn');
const homeBtn = document.getElementById('homeBtn');
const projectBtn = document.getElementById('projectBtn');

let currentPath = '';
let selectedFile = null;
let targetInputId = null;

// Open file browser when Browse button clicked
document.querySelectorAll('.browse-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        targetInputId = e.target.dataset.target;
        openFileBrowser();
    });
});

// Close modal handlers
modalClose.addEventListener('click', closeFileBrowser);
cancelBrowseBtn.addEventListener('click', closeFileBrowser);

// Click outside modal to close
fileBrowserModal.addEventListener('click', (e) => {
    if (e.target === fileBrowserModal) {
        closeFileBrowser();
    }
});

// Navigation buttons
parentDirBtn.addEventListener('click', () => {
    if (currentPath) {
        browseDirectory(currentPath, true);
    }
});

homeBtn.addEventListener('click', () => {
    browseDirectory(null); // Will use HOME from server
});

projectBtn.addEventListener('click', () => {
    browseDirectory('../'); // Navigate to project root
});

// Select file button
selectFileBtn.addEventListener('click', () => {
    if (selectedFile && targetInputId) {
        document.getElementById(targetInputId).value = selectedFile;
        closeFileBrowser();
    }
});

function openFileBrowser() {
    fileBrowserModal.classList.add('show');
    selectedFile = null;
    selectFileBtn.disabled = true;
    selectedPathSpan.textContent = 'None';
    browseDirectory(null); // Start at HOME
}

function closeFileBrowser() {
    fileBrowserModal.classList.remove('show');
    selectedFile = null;
    targetInputId = null;
}

async function browseDirectory(path = null, useParent = false) {
    fileList.innerHTML = '<div class="loading-spinner">Loading...</div>';

    try {
        const url = new URL('/api/browse', window.location.origin);
        if (path) {
            url.searchParams.set('path', path);
        }

        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to browse directory');
        }

        currentPath = useParent ? data.parentPath : data.currentPath;
        breadcrumb.textContent = currentPath;

        // Render file list
        renderFileList(data.items);

    } catch (error) {
        console.error('Browse error:', error);
        fileList.innerHTML = `<div class="loading-spinner" style="color: red;">Error: ${error.message}</div>`;
    }
}

function renderFileList(items) {
    if (items.length === 0) {
        fileList.innerHTML = '<div class="loading-spinner">Empty directory</div>';
        return;
    }

    fileList.innerHTML = '';

    items.forEach(item => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        if (item.isDirectory) {
            fileItem.classList.add('directory');
        }

        const icon = document.createElement('span');
        icon.className = 'file-icon';
        icon.textContent = item.isDirectory ? '📁' : (item.isCheckpoint ? '📄' : '📃');

        const name = document.createElement('span');
        name.className = 'file-name';
        name.textContent = item.name;

        const size = document.createElement('span');
        size.className = 'file-size';
        if (!item.isDirectory && item.size) {
            size.textContent = formatFileSize(item.size);
        }

        fileItem.appendChild(icon);
        fileItem.appendChild(name);
        fileItem.appendChild(size);

        // Click handler
        fileItem.addEventListener('click', () => {
            if (item.isDirectory) {
                // Navigate into directory
                browseDirectory(item.path);
            } else {
                // Select file
                selectFile(item.path, fileItem);
            }
        });

        fileList.appendChild(fileItem);
    });
}

function selectFile(filePath, element) {
    // Remove previous selection
    document.querySelectorAll('.file-item.selected').forEach(el => {
        el.classList.remove('selected');
    });

    // Mark as selected
    element.classList.add('selected');
    selectedFile = filePath;
    selectedPathSpan.textContent = filePath;
    selectFileBtn.disabled = false;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ===== PROCESS VISUALIZATION =====

async function setupProcessVisualization(frameUrls) {
    // Preload all frames
    animationFrames = [];
    currentFrame = 0;
    isPlaying = false;

    imageContainer.innerHTML = '<div class="placeholder"><p>Loading frames...</p></div>';

    // Preload images
    const loadPromises = frameUrls.map((url, index) => {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                animationFrames[index] = img;
                resolve();
            };
            img.onerror = reject;
            img.src = url + '?t=' + Date.now();
        });
    });

    try {
        await Promise.all(loadPromises);

        // Setup slider
        frameSlider.max = animationFrames.length - 1;
        frameSlider.value = 0;

        // Show first frame
        updateFrame();

        // Show animation controls
        animationControls.style.display = 'block';

        // Auto-start animation
        startAnimation();
    } catch (error) {
        console.error('Error loading frames:', error);
        showStatus('Failed to load animation frames', 'error');
    }
}

function updateFrame() {
    if (animationFrames.length === 0) return;

    // Ensure frame is in bounds
    currentFrame = Math.max(0, Math.min(currentFrame, animationFrames.length - 1));

    // Update image
    const img = animationFrames[currentFrame];
    imageContainer.innerHTML = '';
    imageContainer.appendChild(img.cloneNode());

    // Update controls
    frameSlider.value = currentFrame;
    frameCounter.textContent = `Frame: ${currentFrame + 1} / ${animationFrames.length}`;
}

function startAnimation() {
    if (animationFrames.length === 0) return;

    isPlaying = true;
    playPauseBtn.textContent = '⏸ Pause';

    const fps = parseInt(speedControl.value);
    const interval = 1000 / fps;

    animationInterval = setInterval(() => {
        currentFrame++;
        if (currentFrame >= animationFrames.length) {
            currentFrame = 0; // Loop
        }
        updateFrame();
    }, interval);
}

function stopAnimation() {
    isPlaying = false;
    playPauseBtn.textContent = '▶ Play';

    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
    }
}

function togglePlayPause() {
    if (isPlaying) {
        stopAnimation();
    } else {
        startAnimation();
    }
}

function restartAnimation() {
    stopAnimation();
    currentFrame = 0;
    updateFrame();
}
