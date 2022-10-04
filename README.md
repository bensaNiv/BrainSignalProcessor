
# Signal-Behavioral Analysis Tool

This tool was built for the Citri-Lab - Edmond and Lily Safra Brain Research Center, The Hebrew University in Jerusalem.
The purpose of this program is to process the brain signal of a mouse and compare it to its behavioral characterization. The behavioral analysis is performed in an outside separate tool.

## Authors

- [@bensaNiv](https://github.com/bensaNiv)


## Project Structure

The tool is built in a specific hierarchical way, In each experiment folder should be:
- mice folders: 
  * The 470 signal of the mouse trial.
  * The 415 isosbestic signal of the mouse trial.
  * 'tags' Folder:
      A single .pkl file or single .csv/.xlsx file that contains the behavioral tag for that trial.
- mice_key.csv file for general comparison (More explanations below).
- 'plot' folder  - Output folder for the plots from the experiment processing.


## Install Locally

Clone the project

```bash
  git clone https://github.com/bensaNiv/StriatumContext
```

Go to the project directory

```bash
  cd BrainSignalProcessor

```

Open the 'main.py' file and paste your requested path to experiment.

## Usage

In the file 'main.py' , choose an experiment directory to process

```python
exp_folder = r"path_to_dir"
```
Choose does the tags for behavioral set were Automatic tagged or manual tagged.
```python
  '''
  manual_tagged == -1: tagged manualy - Find all shows that contains act letters, == 0 tagged by Stereo,
  == 1 tagged manualy - find only specific shows of this act
  '''
```
Run the process_experiment() function following by onset/offset analysis.

### General comparison

For comparing between 2 different parametrs that are not the deafult left-right hemisphere comparsionL
* Attach a 'mice_key.csv' file to the 'exp_folder' that has the format:

| name            | 2G   | 3G   |
| --------------- | -----|------|
| CG$ms$ | param1 | param2
| CG$ms$ | param1 | param 2

* make sure that the mice's names are in the format CG$ms$ so the names can be detected.

When running the processor, the comparison will happen based on the param1/param2 parameters.
If no 'mive_key.csv' file was given then the comparison will be based on the right/left hemisphere.
##  Support

For support, email bensalniv@gmail.com. 


## Acknowledgements

 - [Citri Lab](https://www.citrilab.com/)
