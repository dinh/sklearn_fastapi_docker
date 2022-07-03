build:
	docker-compose build
up:
	docker-compose up -d
restart:
	make build
	make up
down:
	docker-compose stop
down:
	docker-compose down --volumes