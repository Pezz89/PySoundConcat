.. _descriptor_defs:

Audio Descriptor Definitions
============================
This section describes the audio descriptors used for analysing chacteristics
of the audio files. Each descriptor is used for measuring a specific
characteristic and multiple descriptors are combined to match grains based on
the amalgamation of these measurements. For example, Using the F0 and RMS
descriptors would match audio based on it's pitch and energy.

Centroid
~~~~~~~~~~~~~~~~~
The temporal centroid is a measure of the center of gravity of a signal. It is
used to determine the central point of a signal's amplitude and is calculated
as:

.. math::
    C(n) = \frac{\sum_{i=i_s(n)}^{i_e(n)}(i-i_s(n)) \cdot x(i)}{\sum_{i=i_s(n)}^{i_e(n)} \cdot x(n)}

Ref: :cite:`lerch2012itaca`

F0 (Pitch detection)
~~~~~~~~~~~~~~~~~~~~
An important feature of any periodic audio is it's pitch. Pitch is defined as
the perceived frequency of the signal. In order to determine the pitch of a
periodic signal, the fundamental frequency (:math:`f0`) is estimated. There are
many methods developed for estimating the :math:`f0` of a signal. This program
uses the autocorrelation method. This method was chosen for it's simplicity and
reasonable versatility for a wide range of signals.

The f0 is calculated by first calculating the autocorellation of the signal
defined as:

.. math::
    R_n(m) = \sum_{i=i_s(n)}^{i_e(n)} x(i) x(i-m)

Then normalizing:

.. math::
    \Gamma_n(m) = \frac{R_n(m)}{\sqrt{\sum_{i=i_s(n)}^{i_e(n)}x(i)^2 \sum_{i=i_s(n)}^{i_e(n)}x(i-m)^2}}

The fundamental period of the signal is then calculated as the point between
:math:`T_{min}` and :math:`T_{max}` at which the correlated signal most closely matches the
original. :math:`T_{min}` and :math:`T_{max}` are defined as the minimum and maximum values of
the fundamental period.

.. math::
    y = arg\,max_{T_{min} \leq m \leq T_{max}} \{\Gamma_i(m)\}

In order to improve the accuracy of peak detection, parabolic interpolation is
used to estimate the peak's location with greater accuracy by using the peak
correlation and it's two closes neighbour's values to estimate the fractional
peak value.

The method for parabolic interpolation is defined as:

.. math::
    \Gamma_0^n = \frac{1}{2} \cdot \frac{\alpha - \gamma}{\alpha - 2\beta + \gamma} + y

    &\text{Where:} \\
    &\alpha = \gamma(y-1) \\
    &\beta = \gamma(y) \\
    &\gamma = \gamma(y+1) \\
Ref: :cite:`smith2011sasp`

From this, the fundamental period the frequency is then calculated as:

.. math::
    f_0^n = \frac{1}{T_0^n}

Ref: :cite:`itaa2014`


FFT
~~~
The FFT algorithm is an optimized algorithm for computing the Short Time
Fourier Transform for windows of a signal. The full description of this
transform is outside the scope of this project, however it should be understood
that this analysis provides a description of the spectral content of a windowed
signal. By applying the transform, a number of bins of size :math:`K` are
calculated that detail the sine and cosine apmlitudes required to reconstruct
the signal. The calculation of the STFT is defined as:

.. math::
    X(k,n) = \sum_{i=i_s(n)}^{i_e(n)} x(i) \exp{\Big(-jk \cdot (i -
    i_s(n))\frac{2\pi}{K}\Big)}

Ref: :cite:`lerch2012itaca`

Harmonic Ratio
~~~~~~~~~~~~~~
The harmonic ratio can be used to differentiate between noisy and periodic
signals. higher values suggest that the signal is more periodic (such as a sine
wave) and lower values represent less periodicity. This can be used as a form
of confidence measure in determining the validity of F0 values. it is
calculated as part of the F0 estimation algorithm as:

.. math::
    HR(n) = max_{T_{min} \leq m \leq T_{max}}{\{T_n(m)\}}

Ref: :cite:`lerch2012itaca`

Kurtosis
~~~~~~~~~~~~~~~~~
Temporal kurtosis is used for measuring the flatness of the signal. Lower
values indicate a flatter distribution and positive values indicate a more
"peaky" distribution. Kurtosis is calculated as:

.. math::
    TK(n)=\frac{1}{\sigma_x^4(n) \cdot K}\sum_{i=i_s(n)}^{i_e(n)}\Big(x(i)-\mu_x(n)\Big)^4-3

Ref: :cite:`lerch2012itaca`

Peak Amplitude
~~~~~~~~~~~~~~
Peak amplitude measures the highest peak in the absoulte signal. it is
calculated as:

.. math::
    P(n) = \max_{i_s(n) \leq i \leq i_e(n)}\{\left|x(i)\right|\}

RMS
~~~
The perceived loudness of a signal is an important feature as it can be related
to the dynamics of the signal.  RMS is used as a measure of sound intensity and
is used for distinguishing between loud and quiet audio. It is calculated as:

.. math::
    RMS(n) = \sqrt{\frac{1}{K} \sum_{i=i_s(n)}^{i_e(n)} x(i)^2}

Other methods that take the human perception of loudness into account may
provide more perceptually relevant results. However the RMS measurement
produced acceptable results for this application.

Ref: :cite:`lerch2012itaca`

Spectral Centroid
~~~~~~~~~~~~~~~~~
The spectral centroid measure the center of gravity accross frequency bins to
determine the central point accross the spectral content of the frame. High
value sindicate that the spectral content is centered in higher frequencies and
lower value indicate a lower center. The spectral centroid is calculated as:

.. math::
    SC(n) = \frac{\sum_{k=0}^{K/2-1} k \cdot | X(k,n) | ^2}{\sum_{k=0}^{K/2-1} | X(k,n) | ^2}

The result is the sum of magnitudes, weighted by their index, normalized by the
unweighted sum.

Ref: :cite:`lerch2012itaca`

Spectral Crest Factor
~~~~~~~~~~~~~~~~~~~~~
The spectral crest factor can be used as a mesure of tonalness of the signal.
it is calculated by taking the maximum magnitude and dividing by the sum of
magnitudes.
This differntiates between flat spectrums and sinusoidal spectrums. (low values
represnting the former and high values representing the latter.)

.. math::
    SCF = \frac{ \max_{0 \leq k \leq K/2-1} \{| X(k,n) | \}}{\sum_{k=0}^{K/2-1} | X(k,n) | }

Ref: :cite:`lerch2012itaca`

Spectral Flatness
~~~~~~~~~~~~~~~~~
Defined as the ratio between the geometric and arithmetic mean of the magnitude
spectrum, spectral flatness indicates the noisiness of a signal. Higher values
indicate a flatter spectrum (suggesting a noisy signal) as opposed to lower
values that represent a more tonal signal. Spectral flatness is calculated as:

.. math::
    TFl(n) = \frac{\sqrt[K/2]{\prod_{k=0}^{K/2-1} | X(k,n) | }}{2/K \cdot
    \sum_{k=0}^{K/2-1} | X(k,n) | }

Ref: :cite:`lerch2012itaca`

Spectral Flux
~~~~~~~~~~~~~
Spectral flux is a measure of change between consecutive frames. It calculates
the average difference between frames to differentiate between adjacent frames
that are largely dissimilar (suggesting a non-stationary section of signal) and
similiar frames (that suggests a steady state signal). It is calculated as:

.. math::
    SF(n) = \frac{\sqrt{\sum_{k=0}^{K/2-1} \Big( | X(k,n) | - | X(k,n-1) | \Big)^2
    }}{K/2}

Ref: :cite:`lerch2012itaca`

Spectral Spread
~~~~~~~~~~~~~~~
Spectral spread is a measurement of the concentration of magnitudes around the
spectral centroid. This description relates to the spectral shape of the signal
and is associated with perceptions of timbre. It is calculated as:

.. math::
    SS(n) = \sqrt{\frac{\sum_{k=0}^{K/2-1} \Big(k-SC(n)\Big)^2 \cdot | X(k,n)
    | ^2}{\sum_{k=0}^{K/2-1} | X(k,n) | ^2}}

Ref: :cite:`lerch2012itaca`

Variance
~~~~~~~~
The variance of a signal measures it's spread around the signal's arithmetic
mean. It is used in the calculation of Kurtosis and is calculated as:

.. math::
    \sigma_x^2 = \frac{1}{K} \sum_{i=i_s(n)}^{i_e(n)}(x(i) - \mu_x(n))^2    

Ref: :cite:`lerch2012itaca`

Zero-Crossing
~~~~~~~~~~~~~
The zero-crossing rate counts the number of times a signal's value changes from
positive to negative in a frame. it is relevant to determining the noisiness of
a signal, as noisy signals will pass from positive to negative more frequenctly
than period signals. It is calculated as:

.. math::
    Z(n) = \frac{1}{2K} \sum_{i=i_s(n)}^{i_e(n)} | sgn[x(i)] - sgn[x(i-1)] |

    \text{Where the sgn function is defined as:}
    
    sgn[x_i(n)] = \left\{
                \begin{array}{ll}
                1, x(i) \geq 0\\
                -1, x(i) < 0
                \end{array}
              \right.

Ref: :cite:`itaa2014`

List of Symbols
~~~~~~~~~~~~~~~

====================  ================================================
Symbol                  Meaning
====================  ================================================
:math:`C`               Centroid
:math:`f`               frequency
:math:`\Gamma`          Normalized autocorrelation
:math:`HR`              Harmonic ratio
:math:`i`               Sample index
:math:`i_e`             End index of frame
:math:`i_s`             Start index of frame
:math:`K`               Size of frame
:math:`m`               Correlation time lag
:math:`\mu_x`           Arithmetic Mean
:math:`n`               Frame index
:math:`P`               Peak amplitude
:math:`R`               Autocorrelation of signal
:math:`RMS`             Root Mean Square
:math:`\sigma_x^2`      Variance
:math:`SC`              Spectral centroid
:math:`SCF`             Spectral crest factor
:math:`SF`              Spectral flux
:math:`SS`              Spectral spread
:math:`TK`              Kurtosis
:math:`TFl`              Spectral flatness
:math:`x`               Audio signal
:math:`X(k,n)`          STFT of current frame
:math:`Z`               Zero-crossing rate
====================  ================================================
.. bibliography:: refs.bib
