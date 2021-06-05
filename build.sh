# ./build.sh georgesoteriou/sr:0.1
TAG="${1:-georgesoteriou/sr:0.1}"
cd frontend; npm install; npm run build; cd ..;
docker build -t $TAG .
# docker push georgesoteriou/sr:0.1