# Setup Instructions

## Step 1: Install Node.js

This web application requires Node.js to run the server.

### On macOS:

```bash
# Using Homebrew
brew install node

# Or download from https://nodejs.org/
```

### On Linux:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nodejs npm

# CentOS/RHEL
sudo yum install nodejs npm
```

### Verify Installation:

```bash
node --version   # Should show v14.x or higher
npm --version    # Should show 6.x or higher
```

## Step 2: Install Dependencies

Navigate to the webapp directory and install Node.js packages:

```bash
cd ~/Dropbox/coding/ddpm-mnist/webapp
npm install
```

This will install:
- `express` - Web server framework
- `multer` - File upload handling
- `nodemon` - Development server with auto-reload

## Step 3: Verify Python Setup

Make sure Python and PyTorch are available:

```bash
python --version    # Python 3.7+
python -c "import torch; print(torch.__version__)"
```

## Step 4: Train or Locate Models

You need trained models before you can generate images. Either:

**Option A: Train models yourself**

```bash
# Train autoencoder
cd ~/Dropbox/coding/ddpm-mnist/mnist_ae
python train.py --model conv --epochs 100 --generate-images

# Train latent diffusion
cd ~/Dropbox/coding/ddpm-mnist/mnist_latent_diffusion
python train.py --ae-ckpt /path/to/autoencoder/latest.pt --epochs 30 --generate-images
```

**Option B: Use existing checkpoints**

If you have pre-trained models, note their paths. You'll need:
1. A latent diffusion checkpoint (from `mnist_latent_diffusion/`)
2. An autoencoder checkpoint (from `mnist_ae/`)

## Step 5: Start the Server

```bash
cd ~/Dropbox/coding/ddpm-mnist/webapp
npm start
```

You should see:
```
Server running on http://localhost:3000
Ready to generate MNIST digits!
```

## Step 6: Open the Web Interface

Open your browser to:
```
http://localhost:3000
```

You should see the MNIST Latent Diffusion Generator interface!

## Quick Test

Once the server is running, try generating images:

1. Enter a diffusion checkpoint path (or select from dropdown after clicking "Refresh Checkpoints")
2. Optionally enter an autoencoder checkpoint path
3. Set number of images to 16
4. Click "Generate Images"
5. Wait 30-60 seconds
6. Generated images should appear!

## Troubleshooting

### "Cannot find module 'express'"
- Run `npm install` in the webapp directory

### "Python command not found"
- Ensure Python is in your PATH
- On SLURM systems, you may need to activate your virtualenv first

### "No checkpoints found"
- Train models or manually enter checkpoint paths
- Checkpoints should be in `outputs/checkpoints/` or `$SCRATCH/checkpoints/`

### Generation hangs
- Check server console for Python errors
- Verify checkpoint paths are correct
- Ensure GPU/CPU is available for inference

### Port 3000 already in use
- Change PORT in server.js or set environment variable:
  ```bash
  PORT=3001 npm start
  ```

## Development Mode

For development with auto-reload on file changes:

```bash
npm run dev
```

This uses `nodemon` to automatically restart the server when you modify `server.js`.

## Next Steps

- Customize the UI in `public/index.html` and `public/style.css`
- Add more generation options (temperature, sampling steps, etc.)
- Implement real-time progress updates with WebSockets
- Add image gallery to browse previously generated images
- Deploy to a cloud platform for remote access
