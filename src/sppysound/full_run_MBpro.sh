set -euo pipefail
./create_database.py /Volumes/Storage/AudioDatabases/Viola ~/AllDatabases/AnalysedAudioDatabases/Viola3 --reanalyse
./create_database.py /Volumes/Storage/AudioDatabases/Vocal_examples ~/AllDatabases/AnalysedAudioDatabases/Vocal_examples --reanalyse
./run_matching.py ~/AllDatabases/AnalysedAudioDatabases/Viola3 ~/AllDatabases/AnalysedAudioDatabases/Vocal_examples ~/AllDatabases/OutputDatabases/TestOutput --rematch
./synthesize_output.py ~/AllDatabases/AnalysedAudioDatabases/Viola3 ~/AllDatabases/OutputDatabases/TestOutput ~/AllDatabases/AnalysedAudioDatabases/Vocal_examples
