# shellcheck disable=SC1114
#!/usr/bin/sh

cd frontend
 docker image build -t churn-frontend:latest .

cd ../backend
docker image build -t churn-api:latest .