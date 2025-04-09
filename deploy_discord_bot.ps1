# Deploying Discord Bot to Google Cloud Run
Write-Host "Deploying Discord Bot to Google Cloud Run..."

# Set your project ID
$PROJECT_ID = "eddies-discord-bot"

# Set the current project
Write-Host "Setting current project to $PROJECT_ID..."
$result = gcloud config set project $PROJECT_ID
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to set project"
    exit 1
}

# Check if .env file exists
if (-not (Test-Path .env)) {
    Write-Error "Error: .env file not found!"
    Write-Host "Please create a .env file with your DISCORD_TOKEN"
    exit 1
}

# Read DISCORD_TOKEN from .env file
Write-Host "Reading Discord token from .env file..."
$DISCORD_TOKEN = ""
Get-Content .env | ForEach-Object {
    if ($_ -match "^DISCORD_TOKEN=(.*)$") {
        $DISCORD_TOKEN = $matches[1]
    }
}

# Verify if token was read successfully
if ([string]::IsNullOrEmpty($DISCORD_TOKEN)) {
    Write-Error "Error: Could not read DISCORD_TOKEN from .env file"
    Write-Host "Please ensure your .env file contains a line like: DISCORD_TOKEN=your-token-here"
    exit 1
}

Write-Host "Token read successfully: $DISCORD_TOKEN"

# Enable required APIs
Write-Host "Enabling required APIs..."
$result = gcloud services enable cloudbuild.googleapis.com
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to enable Cloud Build API"
    exit 1
}

$result = gcloud services enable run.googleapis.com
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to enable Cloud Run API"
    exit 1
}

# Build and deploy the container
Write-Host "Building and deploying container..."
$result = gcloud builds submit --project $PROJECT_ID --tag "gcr.io/$PROJECT_ID/discord-bot"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to build and submit container"
    exit 1
}

# Deploy to Cloud Run
Write-Host "Deploying to Cloud Run..."
$result = gcloud run deploy discord-bot `
    --image "gcr.io/$PROJECT_ID/discord-bot" `
    --platform managed `
    --region us-central1 `
    --allow-unauthenticated `
    --project $PROJECT_ID `
    --set-env-vars "DISCORD_TOKEN=$DISCORD_TOKEN"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to deploy to Cloud Run"
    exit 1
}

Write-Host "Deployment complete!" 