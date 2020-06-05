## setup
0.  Install Python 3.7.4+ (Anything 3+ probably works)  
1.  Clone git
2.  Create virtual enironment and activate it
3.  Install python packages
4.  Run `main.py`
```
python3 -V

python3 -m venv .venv

.venv\Scripts\activate.bat [WINDOWS]
source .venv/bin/activate  [LINUX]

pip install -r requirements.txt

python main.py

```

## file structure
```
root
 | - main.py
 | - blackbox.py
 | - config.ini
 | - audio
 |    | - RAVDESS-DB
 |    |    | - 1-1-1.wav
 |    |    | - ...
 |    | - Toronto-DB
 |    | - Database-DB
 | - cache
      | - database.npy (automatically created)
```
## blackbox doc

available functions and values
```
b = BlackBox(config=)     # Create BlackBox from a config file

# FUNCTIONS
b.loadData(database=)     # Load database from /audio directory
b.clearCache()            # Delete all existing cached databases

b.train()                 # Train model with loaded databases

b.predict()               # Predict test data (test_split > 0.0)
b.predict(f=)             # Predict single file (test_split >= 0.0)

b.updateConfig(section, option, value, wipe=False)  # Update a single config parameter, wipe clears loaded databases
b.getConfigValue(section, option)                   # Get single config parameter value
b.printConfig()                                     # Dump complete config to console

# VALUES    
b.accuracy                # BlackBox accuracy after predicting using train/test split  (test_split > 0.0)       
b.databases               # Databases configured in config
b.databases_loaded        # All databases that have been loaded
b.emotions                # Emotions configured in config
```

## config structure
```
[DEBUG]
verbose=0
prediction_details=0

[KNN]
k=3             
q=1             
w=distance  
n_jobs=-1

[DATA]
test_size=0.25  
random_state=None 

[POLYFIT]
snippet_size=30 
poly_degree=10 

[DATABASES]
RAVDESS-DB
Toronto-DB
;beaEmoV-DB
;jenieEmoV-DB
;joshEmoV-DB
;samEmoV-DB

[EMOTIONS]
;["happy", "suprise", "angry", "sad", "fear", "disgust", "neutral"]
happy
suprise
angry
sad
fear
disgust
neutral
```

## audio file name
```
1-1-1.wav
| | |_ id
| |___ male [1] or female [2]
|_____ emotion ("happy", "suprise", "angry", "sad", "fear", "disgust", "neutral")
```
