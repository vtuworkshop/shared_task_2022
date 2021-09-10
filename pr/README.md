# Task 1: Punctuation Restoration (PR)

### About the task
Punctuation restoration is a common post-processing problem for automatic speech recognition systems. It restores boundaries of sentences, clauses from ASR text. Restoring punctuation enables the use of existing tools and methods on ASR text. It also signficantly improve the readability of the text. 

Here is an example of input/output of an end-to-end punctuation restoration system:

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

Toward this end, this task is modeled as a sequence labeling problem at token level in which a token is labeled with the punctuation that follow the token.

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

### Data example
```
ok	  COMMA
let's	O
see	  PERIOD
so	  COMMA
I'm	  O
going	O
to	  O
call	O
this	PERIOD
```

### Evaluation

The system's performance is evaluated using the standard precision, recall, f-score (micro) on token-level.

