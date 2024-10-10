**K-DATA.co.uk deployment repositoryf**

* db.sqlite3 \- sqlite file for all models  
* gunicorn\_config.py \- Gunicorn config for EC2  
* manage.py \- Django top level file  
* requirements.txt \- Python environment requirements  
* reset\_db.py \- Script that resets all databases  
* kdata \- Django files  
  * asgi.py \- asgi configuration for Django  
  * settings.py \- Django settings  
  * urls.py \- Django urls  
  * views.py \- Django views  
  * wsgi.py wsgi configuration for Django  
* kdata\_tf \- Tensorflow files  
  * Dockerfile \- File to initialise Docker containers for training (lifters)  
  * gameState\_vect.py \- Vectorized Poker engine  
  * kdata\_os\_lib.py \- Library for lifter operations  
  * kdata\_proc\_lib.py \- Processing library for game results  
  * kdata\_tf\_lib.py \- Library for Neural Network operations  
  * models.py \- Django models  
  * modlib.py \- Neural Network model generation library  
* lifter-out \-   
  * game\_lifter.py \- Lifter script for battling two networks against each other  
  * train\_lifter.py \- Lifter script for training  
* poker\_royale \- Django app folder  
  * models.py \- Neural Network representative sql model

