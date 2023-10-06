# Offline Skill Graph

Offline Skill Graph if a framework for applying Offline RL in real world applications. The framework connects Offline RL skills in a Graph which combined with a State-to-Skill network allows an agent to plan a solution to a task from any starting point in its environment.
 
This repo containes the code files used for the Offline Skill Graph (OSG) research paper that can be found [here](https://arxiv.org/abs/2306.13630).

### Installation

In order to run the code, two folders should be installed locally using ```pip install -e .```. The ```Panda_Robot_Sim folder``` and the ```Planning``` folder under ```scripts```.

### Usage

There are two test scripts available in the ```scripts``` forlder. ```Test_full.py``` runs a test for the full pipeline as described in the paper, to visualize the tests make sure ```gui``` variable is set to ```True```.

```Test_state2skill.py``` runs the tests for the State-to-Skill network alone, as also described in the paper. ```gui``` variable should be set to ```True``` here aswell if visualization is required.

It is also possible to train a new State-to-Skill network using ```State2Skill.py```.

Use ```main.py``` in TD3_BC folder to learn new skills using new datasets or replicate the existing skills using this works data files. 

### Data files

The Data files used to learn the skills used in the paper can be found [here](https://drive.google.com/drive/folders/1eSHySuRxk0WSG6xuclHlvJX7xDZ8Sp0u?usp=sharing)

For ease of use the datasets folder should be place under ```Panda_Robot_Sim/scripts/data_collection'''.

### TD3+BC
TD3+BC was used to learn all skills used in this work, The code for using this algorithm is includded in this repo together with the trained models used in this works experiments. TD3+BC's original repo can be found [here](https://github.com/sfujim/TD3_BC) with more information and link to the paper.