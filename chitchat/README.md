# Task 3: Chitchat Detection

### About the task

One of the issues for the live-stream videos is that the streamer 
might get involved with off-topic discussions with the audience, 
hence diverging from the main topic and rendering the video uninformative. 
These off-topic sections, which we call them Chitchat, 
could introduce considerable challenges for the downstream applications 
on transcript processing.

This task aims to detect chitchat sentences in the transcript. 
It is modeled as a sentiment analysis problem at sentence level. 
Participants will be provided human-annotated training and development datasets.


### Data

Participants will be provided human-annotated training and development datasets.
The provided data files are in ConLL format (tab-separated). Each line contains a single sentence with its label.
Each file contains transcript of a video clip of 5 minutes.
The data comes with two labels as followed:

```
CHITCHAT
O
```


### Evaluation
The system's performance is evaluated using the standard accuracy on sentence-level.
