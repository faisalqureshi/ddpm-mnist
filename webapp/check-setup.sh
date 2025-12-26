#!/bin/bash
# Setup verification script for MNIST Diffusion Web App

echo "========================================="
echo "MNIST Diffusion Web App - Setup Check"
echo "========================================="
echo ""

# Check Node.js
echo "Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✓ Node.js installed: $NODE_VERSION"
else
    echo "✗ Node.js NOT found"
    echo "  Install from: https://nodejs.org/"
    exit 1
fi
echo ""

# Check npm
echo "Checking npm..."
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo "✓ npm installed: $NPM_VERSION"
else
    echo "✗ npm NOT found"
    exit 1
fi
echo ""

# Check Python
echo "Checking Python..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo "✓ Python installed: $PYTHON_VERSION"
else
    echo "✗ Python NOT found"
    exit 1
fi
echo ""

# Check PyTorch
echo "Checking PyTorch..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch installed: $TORCH_VERSION"
else
    echo "✗ PyTorch NOT found"
    echo "  Install with: pip install torch torchvision"
fi
echo ""

# Check node_modules
echo "Checking dependencies..."
if [ -d "node_modules" ]; then
    echo "✓ Node.js dependencies installed"
else
    echo "✗ Node.js dependencies NOT installed"
    echo "  Run: npm install"
fi
echo ""

# Check for checkpoints
echo "Checking for model checkpoints..."
CHECKPOINT_DIRS=(
    "../outputs/checkpoints"
    "$SCRATCH/checkpoints"
)

FOUND_CHECKPOINTS=0
for dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        CKPT_COUNT=$(find "$dir" -name "latest.pt" 2>/dev/null | wc -l)
        if [ $CKPT_COUNT -gt 0 ]; then
            echo "✓ Found $CKPT_COUNT checkpoint(s) in $dir"
            FOUND_CHECKPOINTS=1
        fi
    fi
done

if [ $FOUND_CHECKPOINTS -eq 0 ]; then
    echo "⚠ No checkpoints found"
    echo "  You can still run the app and specify checkpoint paths manually"
fi
echo ""

# Check generated directory
echo "Checking generated directory..."
if [ -d "generated" ]; then
    echo "✓ Generated directory exists"
else
    echo "⚠ Creating generated directory..."
    mkdir -p generated
    echo "✓ Created generated directory"
fi
echo ""

echo "========================================="
echo "Setup check complete!"
echo "========================================="
echo ""
echo "To start the server, run:"
echo "  npm start"
echo ""
echo "Then open your browser to:"
echo "  http://localhost:3000"
echo ""
