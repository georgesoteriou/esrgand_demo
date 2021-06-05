# Run for Dev

## Start Server
```
$ pip install -r requirements.txt
$ python server.py
```
## Start frontend
```
$ cd frontend
$ npm install
$ npm run serve
```

# Build and deploy
```
$ ./build.sh
$ docker push georgesoteriou/sr:0.1
```

# Pull Docker and run

## To run with Nvidea:
```
docker run -d --name='SR' -e 'TIMEOUT'='180' -p '5000:5000/tcp' -e 'NVIDIA_VISIBLE_DEVICES'='<GPU_ID>' -e 'GPU'='1'  --ipc=host --runtime=nvidia 'georgesoteriou/sr:0.1'
```

## To run for CPU:
```
docker run -d --name='SR' -e 'TIMEOUT'='180' -p '5000:5000/tcp' -e 'GPU'='0' --ipc=host 'georgesoteriou/sr:0.1'
```