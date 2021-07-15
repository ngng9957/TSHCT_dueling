# TSHCT with Dueling DRQN

## Simulation
This repository contains Two-Stage Heterogeneous Centralized Training (TSHCT) with Dueling DRQN algorithm for AI Soccer simulation environment.

The AI Soccer environment is available on the [aisoccer-3d](https://github.com/aisoccer/aisoccer-3d) github repository.
The simulator used for this repository is similar to version 0.1 on the aisoccer-3d repository but has no dribble mode.
To run the code on the version 0.1, adjustment has been committed.

This repository corresponds to a folder, such as "player_rulebased-A" and "player_rulebased-B", which are in "examples" folder in ["aisoccer-3d.zip(tar.gz)"](https://github.com/aisoccer/aisoccer-3d/releases).
After unzip "aisoccer-3d.zip(tar.gz)", download this repository into the "examples" folder.

To run the code, edit the "executable" path in "config.json".
Following files can be "executable" path (the files should be executable in linux).
 - "play.py": Program that controls a team by using trained models(used for proponent team while evaluating).
 - "main.py": Program that controls a team by using models which are being trained by "train.py"(used for proponent team while training).
 - "selfplay.py": Program that controls a team by using trained models which are updated when the number of training episodes reaches a particular number(used for opponent team while training).

To train models,
 1. Edit the "executable" path to "main.py"
 2. Delete "config.pickle"
 3. Run python script "train.py"

Envronment: Ubuntu 16.04, python3.5, torch 1.0.0


## Video Clips
**After 0k iterations**

<img width="60%" src=https://user-images.githubusercontent.com/48238345/125584389-21eb1219-e81c-4298-81e3-f26290002b11.gif>

**After 20k iterations**

<img width="60%" src=https://user-images.githubusercontent.com/48238345/125584411-0ce15a09-4495-42dc-a781-149980d5b9cb.gif>

**After 80k iterations**

<img width="60%" src=https://user-images.githubusercontent.com/48238345/125584416-2ac9746f-af78-403d-9721-70844510a5bf.gif>

**After 200k iterations**

<img width="60%" src=https://user-images.githubusercontent.com/48238345/125584424-995f3ded-4e95-4b84-81f5-4341594a4c73.gif>

**Highlights**

<img width="60%" src=https://user-images.githubusercontent.com/48238345/125584429-e593a3ec-18a0-4024-afbf-6a516ca99ede.gif>
