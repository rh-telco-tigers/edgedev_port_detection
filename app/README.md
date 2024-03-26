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
django-admin startproject ImagePredictor
cd ImagePredictor
python manage.py startapp PredictionApp
```