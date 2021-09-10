# Task 2: Domain Adaptation for Punctuation Restoration (DAPR)

### About the task
This task is modeled similar to task 1. 
However, our focus shifts to exploiting existing transcripting text such as TED talk transcript and movie subtitles
for punctuation restoration of live streaming video transcript.



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

