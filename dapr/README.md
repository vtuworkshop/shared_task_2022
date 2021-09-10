# Task 2: Domain Adaptation for Punctuation Restoration (DAPR)

### About the task
This task is modeled similar to task 1. 
However, our focus shifts to exploiting resources such as movie subtitles, video transcripts, and any other available text for punctuation restoration of live streaming video transcript. Our objective is to develop novel methods that make use of existing data and resources. 

Except the trainining/development/testing data from task 1, the participants are encouraged to exploite any other external resources. Please clearly describe the external data being used in your submission.

### Data
The provided training set is derived from a large collection of English movies subtitles, obtained from [opensubtitles.org](https://opus.nlpl.eu/OpenSubtitles.php). The development and testing sets are human annotated, identical to the development and testing in task 1. 

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

