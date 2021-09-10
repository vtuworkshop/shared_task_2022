# Task 1: Punctuation Restoration (PR)

### About the task
Punctuation restoration is a common post-processing problem for automatic speech recognition systems. It restores boundaries of sentences, clauses from ASR text. For instance:

INPUT:
```
bring it back right so most resourceful i'm not sure what's going to be there
```
OUTPUT:
```
Bring it back.
Right.
So most resourceful, I'm not sure what's going to be there.
```
This task is modeled as a sequence labeling problem at token level. 


### Data

Participants will be provided human-annotated training and development datasets.
The provided data files are in ConLL format (tab-separated). 
The data comes with five labels as followed:

```
PERIOD
COMMA
QUESTION
EXCLAMATION
O
```

### Evaluation

The system's performance is evaluated using the standard precision, recall, f-score (micro) on token-level.

