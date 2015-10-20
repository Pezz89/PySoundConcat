=============================================
                SPPySound
=============================================

SPPySound is a python project that acts as a wrapper to the pysndfile library, adding aditional functionality needed for audio and DSP related projects.
Currently the project is focused on the generation of audio descriptor meta data for sound databases as part of my final year project. 
The project is currently in development but aims to generate a range of useful functions for sound manipulation.
The project is designed with flexability in mind so that new functions and features can be added as needed.
To address this the project has a modular nature and currently consists of 3 main classes for the encapsulation of audio files:

AudioFile
This is the base class for creating an audio file object and has methods for simple processed on the audio file.
Functionality such as converting the file to mono and convenience functions for reading and writing grains of audio are included.

AnalysedAudioFile
Inheriting from the AudioFile class, this encapsulates the audio file and the analyses that can be generated through the seperate audio analysis modules.
This will be used for the processing of analyses of audio files and to group audio files with their analyses.

AudioDatabase
An audio database is a class used to group AnalysedAudioFiles for batch processing. This class will be used for comparison methods to compare audiofiles and their meta data.
