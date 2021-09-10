# Shared task at VTU 2022

## News

Sep 10, 2021: Data release

## Important Dates

Paper submission deadline: Nov 12, 2021

Test result submission deadline: Nov 20, 2021

Test result release: Nov: 27, 2021

Paper acceptance notification: Dec 3, 2021

Camera-ready submission: Dec 10, 2021

VTU workshop at AAAI 2022: Feb 28 - Mar 1, 2022

## Tasks

[Punctuation restoration](https://github.com/vtuworkshop/shared_task_2022/tree/main/pr)

[Domain Adaptation for Punctuation restoration](https://github.com/vtuworkshop/shared_task_2022/tree/main/dapr)

[Chitchat Detection](https://github.com/vtuworkshop/shared_task_2022/tree/main/chitchat)

## Submission

All the submission is done my making pull request to this github repository.
A maximum of three system is allowed for each task.
The last pull request beyond the deadline is considered as the official submission.

The submission has to be organized as follow:

For each task, a subfolder, named "submissions", is already created for submission.
The participant can create a team folder (e.g. ``team-1`` in this example) to store your submissions.
In the team folder, the participants can create upto three system folders, according to the maximum of 3 systems. These folders must be named ``system-1``, ``system-2``, ``system-3``.
In the ``system`` folder, the participant should store the .conll files. These files should follow the same format as the training/developement/testing files (tab-separated). The filename has to be identical to the provided testing input files.


```
submissions/
├── README.md
├── team-1
│   ├── system-1
│   │   └── file.conll
│   ├── system-2
│   └── system-3
└── team-2
```

