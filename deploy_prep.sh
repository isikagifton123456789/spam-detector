#!/bin/bash
# Deployment preparation script

echo "🚀 Preparing Spam Detector for deployment..."

# Check if model files exist
if [ ! -f "logistic_model.pkl" ] || [ ! -f "count_vectorizer.pkl" ]; then
    echo "⚠️  Model files not found. Training model..."
    python train_model.py
else
    echo "✅ Model files found"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📄 Creating .gitignore..."
    cat > .gitignore << EOF
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.pytest_cache/
.DS_Store
.vscode/
*.log
EOF
fi

echo "✅ Deployment preparation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create a GitHub repository"
echo "2. Push your code to GitHub"
echo "3. Connect to your chosen hosting platform"
echo "4. Deploy!"
