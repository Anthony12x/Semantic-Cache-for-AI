Write-Host "Initializing Host-Level AI Infrastructure..." -ForegroundColor Cyan

# 1. Check if Ollama CLI is installed on the host
if (!(Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "FATAL: Ollama is not installed on the host machine. Please install it from https://ollama.com/" -ForegroundColor Red
    exit 1
}

# 2. Check if the daemon is responding
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434" -UseBasicParsing
    Write-Host "Ollama Daemon is running." -ForegroundColor Green
} catch {
    Write-Host "FATAL: Ollama Daemon is not responding. Ensure the background service is running." -ForegroundColor Red
    exit 1
}

# 3. Pull the required models
$MODEL_NAME = "tinyllama"
Write-Host "Pulling model weights for: $MODEL_NAME (This may take a moment)..."
ollama pull $MODEL_NAME

Write-Host "Host infrastructure initialized. You may now run 'docker compose up --build -d'." -ForegroundColor Green