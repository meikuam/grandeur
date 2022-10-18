
cd docker
docker build -t grandeur:dev dev --build-arg UID=$(id -u $(whoami))
docker run -it --gpus=all -d \
    -v /mnt/d/data:/home/user/grandeur/data \
    -v $(readlink -e ../):/home/user/grandeur \
    -p 9223:22 \
    -p 8889:8888 \
    -p 8082:8080 \
    -p 8083:8081 \
    --restart always \
    --name=grandeur-dev2 grandeur:dev
