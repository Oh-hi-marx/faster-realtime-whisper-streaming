This script allows you to send real time audio via tcp to a faster-whisper model, for low latency speech to text transcription. You should run the sendaudio.py on your remote machine and the tcprealtimefasterwhisper.py script on a Nvidia GPU computer. Use small or medium whipser models if turbo is too big for you GPU. You need to change the dest_ip in sendaudio.py to the IP of your GPU computer running thetcprealtimefasterwhisper.py. 
You can adjust vad_threshold to increase the sensitivity to voice activity/suppress background noise.


You need to pip install fasterwhisper and sileroVAD found in the requirements.txt
Make sure you have cuda and cudatoolkit installed before you install fasterwhisper. 

You may need to export your LD path for fasterwhipser to work.
export LD_LIBRARY_PATH=/home/beerspot/miniconda3/envs/openmmlab_clone/lib:$LD_LIBRARY_PATH


