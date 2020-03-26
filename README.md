# RubikCubeRL
## Initial Setup
```
python -m venv .\venv python=3.6  
.\venv\scripts\activate
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