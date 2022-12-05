# Gorilla  

https://app.gorilla.sc/  

For online experiments. Easy builder.  

UZH-LIRI offers licenses and support.  

## Troubleshooting
####  Case \#1. 
\[Q:]
I have converted my audio tasks that worked before with task builder 1 in task builder 2, and I now face some problems: 

1: everytime the task should start (in the preview of the task itself or when run in the experiment tree), there's a "LOADING" appearing for at least 4-6minutes before it starts. What could the source of this problem ? 

2: There's now at the beginning of my audio task "an audio initialisation", which we don't need because we have an audio calibration task before. How can this option be diseable ? 

3: When my different tasks (from task builder 2) are now added to my experimental tree, I can not click on them to modify their caracteristics (e.g. the content of the different manipulation)
This task does not have a content warning

\[A(Gorilla Support):]
I think these issues are caused by one of your empty manipulations - 'list' has an empty options setting, and this could be causing problems. If this manipulation isn't intended to be a dropdown-esque manipulation (have a set of options you can choice from), you should delete the empty options setting.
 
Because your audio zone is bound to this manipulation, and in your binding settings, you haven't set a default, I think Gorilla is trying to load all of your audio files, rather than the specific list subset.
Take another look at this binding and manipulation, to make sure it references your column headings.


## How to randomly apply different lists of stimuli to each participant
(...)  You can create a spreadsheet with multiple columns and then use a node in the spreadsheet node an spreadsheet manipulation to select a different column for each participant

 
# Prolific  

For online recruitment and the paying the participants. Participants will get a prolific ID, they will provide this in the Gorilla experiment. Researchers would not have direct access to identifying information from the prolific ids of the participants .  


 
