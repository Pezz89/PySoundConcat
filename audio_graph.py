import matplotlib.pyplot as plt

def plot_audio(audio_array):
    """
    Plots audio to a graph
    """
    plt.plot(audio_array)
    plt.xlabel("Time (samples)")
    plt.ylabel("sample value")
    plt.show()
