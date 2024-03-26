## Setup virtualenv
```
python3 -m venv venv
source ./venv/bin/activate
```

## Install django
```
pip install django
```

##  Create and Configure Your Django Project
```
django-admin startproject image_predictor
cd image_predictor
python manage.py startapp prediction_app
```

## Run App
```
python manage.py runserver
```