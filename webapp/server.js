const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static('public'));
app.use('/generated', express.static('generated'));

// Helper function to find latest checkpoint
function findLatestCheckpoint(dir) {
    const latestPath = path.join(dir, 'latest.pt');
    if (fs.existsSync(latestPath)) {
        // Resolve symlink
        return fs.realpathSync(latestPath);
    }
    return null;
}

// API endpoint to generate images
app.post('/api/generate', async (req, res) => {
    const {
        numImages = 16,
        digit = null,
        diffusionCkpt = null,
        aeCkpt = null
    } = req.body;

    console.log('Generation request:', { numImages, digit, diffusionCkpt, aeCkpt });

    // Validate inputs
    if (numImages < 1 || numImages > 100) {
        return res.status(400).json({ error: 'Number of images must be between 1 and 100' });
    }

    if (digit !== null && (digit < 0 || digit > 9)) {
        return res.status(400).json({ error: 'Digit must be between 0 and 9' });
    }

    // Generate unique output filename
    const timestamp = Date.now();
    const outputFile = `generated_${timestamp}.png`;
    const outputPath = path.join(__dirname, 'generated', outputFile);

    // Build Python command
    const projectRoot = path.join(__dirname, '..');
    const inferScript = path.join(projectRoot, 'mnist_latent_diffusion', 'infer.py');

    const args = [
        inferScript,
        '--num-images', numImages.toString(),
        '--output', outputPath
    ];

    if (diffusionCkpt) {
        args.push('--ckpt', diffusionCkpt);
    }

    if (aeCkpt) {
        args.push('--ae-ckpt', aeCkpt);
    }

    if (digit !== null) {
        args.push('--digit', digit.toString());
    }

    console.log('Running Python command:', 'python', args.join(' '));

    // Spawn Python process
    const pythonProcess = spawn('python', args, {
        cwd: projectRoot
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
        console.log('Python stdout:', data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
        console.error('Python stderr:', data.toString());
    });

    pythonProcess.on('close', (code) => {
        console.log('Python process exited with code:', code);

        if (code !== 0) {
            return res.status(500).json({
                error: 'Image generation failed',
                details: stderr || stdout
            });
        }

        // Check if file was created
        if (!fs.existsSync(outputPath)) {
            return res.status(500).json({
                error: 'Image file was not created',
                details: stdout
            });
        }

        // Return success with image URL
        res.json({
            success: true,
            imageUrl: `/generated/${outputFile}`,
            output: stdout
        });
    });

    pythonProcess.on('error', (error) => {
        console.error('Failed to start Python process:', error);
        res.status(500).json({
            error: 'Failed to start Python process',
            details: error.message
        });
    });
});

// API endpoint to list available checkpoints
app.get('/api/checkpoints', (req, res) => {
    const projectRoot = path.join(__dirname, '..');
    const checkpoints = {
        diffusion: [],
        autoencoder: []
    };

    // Look for checkpoints in common output locations
    const outputDirs = [
        path.join(projectRoot, 'outputs', 'checkpoints'),
        process.env.SCRATCH ? path.join(process.env.SCRATCH, 'checkpoints') : null
    ].filter(Boolean);

    for (const dir of outputDirs) {
        if (!fs.existsSync(dir)) continue;

        const experiments = fs.readdirSync(dir);

        for (const exp of experiments) {
            const expPath = path.join(dir, exp);
            if (!fs.statSync(expPath).isDirectory()) continue;

            const latestPath = path.join(expPath, 'latest.pt');
            if (fs.existsSync(latestPath)) {
                const realPath = fs.realpathSync(latestPath);

                if (exp.startsWith('lddpm-') || exp.startsWith('latent-diffusion-')) {
                    checkpoints.diffusion.push({
                        name: exp,
                        path: realPath
                    });
                } else if (exp.startsWith('ae-')) {
                    checkpoints.autoencoder.push({
                        name: exp,
                        path: realPath
                    });
                }
            }
        }
    }

    res.json(checkpoints);
});

// API endpoint to browse filesystem for checkpoint files
app.get('/api/browse', (req, res) => {
    const requestedPath = req.query.path || process.env.HOME || '/';
    const projectRoot = path.join(__dirname, '..');

    // Security: restrict to reasonable directories
    const allowedRoots = [
        process.env.HOME,
        projectRoot,
        process.env.SCRATCH,
        '/Users',
        '/home'
    ].filter(Boolean);

    // Normalize and validate the path
    const normalizedPath = path.resolve(requestedPath);
    const isAllowed = allowedRoots.some(root => {
        const normalizedRoot = path.resolve(root);
        return normalizedPath.startsWith(normalizedRoot);
    });

    if (!isAllowed) {
        return res.status(403).json({ error: 'Access denied to this directory' });
    }

    if (!fs.existsSync(normalizedPath)) {
        return res.status(404).json({ error: 'Directory not found' });
    }

    try {
        const stat = fs.statSync(normalizedPath);

        if (!stat.isDirectory()) {
            return res.status(400).json({ error: 'Path is not a directory' });
        }

        const items = fs.readdirSync(normalizedPath);
        const fileList = [];

        for (const item of items) {
            // Skip hidden files
            if (item.startsWith('.')) continue;

            const itemPath = path.join(normalizedPath, item);

            try {
                let itemStat = fs.statSync(itemPath);

                // If it's a symlink, get the real path
                if (itemStat.isSymbolicLink()) {
                    const realPath = fs.realpathSync(itemPath);
                    itemStat = fs.statSync(realPath);
                }

                const isDirectory = itemStat.isDirectory();
                const isCheckpoint = !isDirectory && item.endsWith('.pt');

                fileList.push({
                    name: item,
                    path: itemPath,
                    isDirectory,
                    isCheckpoint,
                    size: isDirectory ? null : itemStat.size,
                    modified: itemStat.mtime
                });
            } catch (err) {
                // Skip files we can't read
                continue;
            }
        }

        // Sort: directories first, then by name
        fileList.sort((a, b) => {
            if (a.isDirectory && !b.isDirectory) return -1;
            if (!a.isDirectory && b.isDirectory) return 1;
            return a.name.localeCompare(b.name);
        });

        res.json({
            currentPath: normalizedPath,
            parentPath: path.dirname(normalizedPath),
            items: fileList
        });
    } catch (error) {
        console.error('Error browsing directory:', error);
        res.status(500).json({ error: 'Failed to read directory', details: error.message });
    }
});

// Cleanup old generated images (older than 1 hour)
function cleanupOldImages() {
    const generatedDir = path.join(__dirname, 'generated');
    const maxAge = 60 * 60 * 1000; // 1 hour

    fs.readdir(generatedDir, (err, files) => {
        if (err) return;

        const now = Date.now();
        files.forEach(file => {
            const filePath = path.join(generatedDir, file);
            fs.stat(filePath, (err, stats) => {
                if (err) return;
                if (now - stats.mtimeMs > maxAge) {
                    fs.unlink(filePath, (err) => {
                        if (!err) console.log('Cleaned up old image:', file);
                    });
                }
            });
        });
    });
}

// Run cleanup every 10 minutes
setInterval(cleanupOldImages, 10 * 60 * 1000);

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log('Ready to generate MNIST digits!');
});
