# shellcheck disable=SC1114
#!/usr/bin/sh

# Use this script if you don't use docker-compose to build the images.
cd frontend
 docker image build -t churn/frontend:latest .

cd ../backend
docker image build -t churn/api:latest .