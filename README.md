# seminar-2022-backend
FastAPI server for subtitle generation.

image
docker build -t poetryi -f Dockerfile .

container
docker create --name poetryc -p 127.0.0.1:8000:8000 poetryi

docker start
docker start poetryc