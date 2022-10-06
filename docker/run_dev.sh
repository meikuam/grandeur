
cd docker
docker build -t grandeur:dev dev --build-arg UID=$(id -u $(whoami))
docker run -it --gpus=all -d \
    -v /mnt/d/data:/home/user/grandeur/data \
    -v $(readlink -e ../):/home/user/grandeur \
    -p 9222:22 \
    -p 8888:8888 \
    -p 8080:8080 \
    -p 8081:8081 \
    --restart always \
    --name=grandeur-dev grandeur:dev
