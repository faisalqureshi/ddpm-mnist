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
