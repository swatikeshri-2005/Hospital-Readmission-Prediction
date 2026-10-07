Write-Host "==============================================="
Write-Host "HOSPITAL READMISSION PREDICTION"
Write-Host "==============================================="

Write-Host ""
Write-Host "[1/4] Downloading dataset..."
python -m src.download_data

if ($LASTEXITCODE -ne 0) {
    Write-Host "Dataset download failed."
    exit 1
}

Write-Host ""
Write-Host "[2/4] Training models..."
python -m src.train

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed."
    exit 1
}

Write-Host ""
Write-Host "[3/4] Evaluating model..."
python -m src.evaluate

if ($LASTEXITCODE -ne 0) {
    Write-Host "Evaluation failed."
    exit 1
}

Write-Host ""
Write-Host "[4/4] Running sample prediction..."
python -m src.predict

Write-Host ""
Write-Host "==============================================="
Write-Host "PIPELINE COMPLETE"
Write-Host "==============================================="

Write-Host ""
Write-Host "To launch Streamlit:"
Write-Host "streamlit run app/app.py"