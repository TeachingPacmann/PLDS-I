# Toxic Comment Classification


## Background
Sosial media menjadi salah satu tempat yang paling aktif di internet. Dimana orang dapat berbagi, berdiskusi dan berinteraksi dengan sesama. Namun didalam sosial media, orang-orang yang toxic akan selalu menyebarkan kebencian, apapun alasannya. Seringkali, komentar toxic ini mengganggu diskusi online. Oleh karena itu kita memiliki tujuan untuk menyaring komentar toxic ini dan menjauhkannya dari diskusi yg "baik" ini.

## Objectives
- Tujuan utama dari task ini adalah melakukan klasifikasi toxic terhadap sebuah komentar.

## Archtecture Overview
![image](https://user-images.githubusercontent.com/46605131/173094283-f1fd62b6-c2e9-459f-a601-d5b3799f4be9.png)

## Architecture in Training Phase.
![image](https://user-images.githubusercontent.com/46605131/173094497-ce13fc3d-32de-49d8-b15c-fdcb02c7e734.png)

## Architecture in Prediction Phase.
![image](https://user-images.githubusercontent.com/46605131/173094571-8739afca-36b0-4560-bc20-cb00e543bfcf.png)

## How to Run
```
cd app/src
gunicorn app.src.api:app --worker-class uvicorn.workers.UvicornWorker
```

## Dataset
- https://drive.google.com/file/d/1f-k3GKloJwZpb9Men9RzZFz20AwBX9qb/view?usp=sharing
- https://drive.google.com/file/d/1bPnGM2cri3hPlYbmzJCeWZEvSPUnH77F/view?usp=sharing -> small


## References
- https://fastapi.tiangolo.com/
