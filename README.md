# Traffic Analyzer API

A FastAPI-based service that analyzes Google Maps traffic using Selenium Grid.  
Supports **batch requests** (up to 20 locations) with a **job queue** for scalable performance.

## Features
- JWT authentication (`/token`)
- Submit multiple locations per batch (`/analyze-batch`)
- Poll job status (`/job/{id}`)
- Cancel Job (`/job/{id}/cancel`)
- Health check endpoint to verify Selenium Grid status (`/health`)
- Results include traffic score, method, screenshot URL
- Queue system ensures VPS stability under load
- SQLite3 database logging, and Caching
- Configurable concurrency
- Tests include:
   - Authentication
   - Batch submission
   - Polling until done
   - Order preservation
   - Mocked parallel jobs

### Requirements
- Python 3.12+
- Docker & Docker Compose
- Selenium Grid (standalone Chrome container or external)


### Setup & Test Localy

#### Setup Python Env & Tests
```bash
python3 -m venv .venv

source .venv/bin/active

pip install -r requirements.txt

pytest -v --disable-warnings
```

#### Setup Selenium Grid Server & API end-point Using Docker
```bash
docker-compose up --build

docker exec -it selenium_hub wget -q -O - http://localhost:4444/status
```

### Run with Docker Compose
```bash
docker-compose up --build
```

### Authentication
Default admin user:
   - username: admin
   - password: 123456
   (Change via ADMIN_PASSWORD env var.)

## Deployment

#### GitHub Actions CI/CD included:
   - Runs tests on PR
   - Deploys to VPS with Docker Compose

#### Secrets required:
   - `VPS_HOST`
   - `VPS_USERNAME`
   - `VPS_SSH_KEY`


## Configuration

#### Environment variables:

   - `JWT_SECRET` → JWT signing key
   - `ADMIN_PASSWORD` → initial admin password
   - `RATE_LIMIT` → e.g. 10/minute
   - `JOBQUEUE_MAX_JOBS` → concurrent jobs (default: 20)
   - `JOBQUEUE_PER_JOB_CONCURRENCY` → Selenium workers per job (default: 4)
   - `SQLITE_DB_FILE` → SQLite file path (default: traffic.db)
   - `SELENIUM_PROXY` → selenium proxy
