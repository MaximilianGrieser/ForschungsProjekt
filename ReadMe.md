# setup
0.  Install Python 3.7.4+ (Anything 3+ probably works)  
1.  Clone git
2.  Create virtual enironment
3.  Install python packages
4.  Run `main.py`
```
python3 -V

python3 -m venv .venv

pip install -r requirements.txt

python main.py

```

# file structure
```
root
 | - main.py
 | - blackbox.py
 | - config.ini
 | - audio
      | - RAVDESS-DB
      |    | - 1-1-1.wav
      |    | - ...
      | - Toronto-DB
      | - Database-DB
```
# file name
```
1-1-1.wav
| | |_ id
| |___ male [1] or female [2]
|_____ emotion ("happy", "suprise", "angry", "sad", "fear", "disgust", "neutral")
```
# config structure
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
