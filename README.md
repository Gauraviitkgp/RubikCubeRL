# RubikCubeRL
## Initial Setup
```
python -m venv .\venv
.\venv\scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Wish to update requirements.txt
```
pip freeze>requirements.txt
```

## Start the training
```
python train.py
```

### For tensorboard
```
tensorboard --logdir tensorboard_training
```