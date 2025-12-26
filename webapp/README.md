# MNIST Latent Diffusion Web Application

A Node.js web interface for generating MNIST digits using trained latent diffusion models.

## Features

- **Interactive UI**: Generate MNIST digits through a web browser
- **Flexible Controls**: Specify number of images, digit type, and checkpoint paths
- **Auto-Detection**: Automatically finds and lists available checkpoints
- **Real-time Generation**: View generated images immediately in the browser
- **Checkpoint Discovery**: Browse trained models from the UI

## Prerequisites

- Node.js (v14 or higher)
- Python 3.x with PyTorch
- Trained diffusion and autoencoder models

## Installation

1. Install Node.js dependencies:
```bash
cd webapp
npm install
```

2. Ensure you have trained models available. The app looks for checkpoints in:
   - `../outputs/checkpoints/`
   - `$SCRATCH/checkpoints/` (if on SLURM)

## Usage

### Start the server

```bash
npm start
```

Or for development with auto-reload:
```bash
npm run dev
```

The server will start on `http://localhost:3000`

### Generate Images

1. Open your browser to `http://localhost:3000`
2. Configure generation settings:
   - **Number of Images**: 1-100 images (default: 16)
   - **Digit to Generate**: Specific digit (0-9) or random
   - **Diffusion Checkpoint**: Path to trained diffusion model
   - **Autoencoder Checkpoint**: Path to autoencoder (auto-detected if not specified)
3. Click "Generate Images"
4. Wait for generation to complete (may take 30-60 seconds)
5. View the generated image grid

### Using Checkpoints

**Option 1: Auto-discovery**
- Click "Refresh Checkpoints" to scan for available models
- Select from dropdowns

**Option 2: Manual paths**
- Enter full paths to checkpoint files
- Useful for checkpoints outside standard directories

## API Endpoints

### POST `/api/generate`

Generate images using the diffusion model.

**Request body:**
```json
{
  "numImages": 16,
  "digit": 5,
  "diffusionCkpt": "/path/to/diffusion/checkpoint.pt",
  "aeCkpt": "/path/to/autoencoder/checkpoint.pt"
}
```

**Response:**
```json
{
  "success": true,
  "imageUrl": "/generated/generated_1234567890.png",
  "output": "Python output logs..."
}
```

### GET `/api/checkpoints`

List available checkpoints.

**Response:**
```json
{
  "diffusion": [
    {"name": "lddpm-...", "path": "/path/to/checkpoint.pt"}
  ],
  "autoencoder": [
    {"name": "ae-conv-...", "path": "/path/to/checkpoint.pt"}
  ]
}
```

## Project Structure

```
webapp/
├── server.js           # Express backend server
├── package.json        # Node.js dependencies
├── public/             # Frontend files
│   ├── index.html      # Main UI
│   ├── style.css       # Styling
│   └── app.js          # Frontend JavaScript
├── generated/          # Temporary storage for generated images
└── README.md           # This file
```

## How It Works

1. **Frontend**: User configures generation settings in the web UI
2. **Backend**: Express server receives request and spawns Python process
3. **Python**: `mnist_latent_diffusion/infer.py` generates images
4. **Response**: Generated image is saved and returned to the frontend
5. **Display**: Image appears in the browser

## Cleanup

Generated images are automatically cleaned up after 1 hour to save disk space.

## Troubleshooting

**No checkpoints found:**
- Ensure you have trained models in `outputs/checkpoints/`
- Manually specify checkpoint paths in the text inputs

**Generation fails:**
- Check that Python environment is set up correctly
- Verify checkpoint paths are correct
- Check server console for Python error messages

**Image doesn't appear:**
- Check browser console for errors
- Verify the generated file exists in `webapp/generated/`
- Try refreshing the page

## Development

The server uses `child_process.spawn()` to call the Python inference script. All Python output (stdout/stderr) is captured and can be viewed in the UI's output log.

To modify generation parameters, edit the API endpoint in `server.js` or add new UI controls in `public/index.html`.

## Production Notes

For production deployment:
- Use a process manager like PM2
- Add authentication if exposing publicly
- Configure CORS if frontend is on different domain
- Implement rate limiting to prevent abuse
- Use environment variables for configuration
