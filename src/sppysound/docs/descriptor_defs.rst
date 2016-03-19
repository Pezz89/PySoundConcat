Audio Descriptor Definitions
============================

Temporal Centroid
~~~~~~~~~~~~~~~~~
The temporal centroid is a measure of the center of gravity of a signal. It is
used to determine the central point of a signals amplitude and is calculated
as:

.. math::
    vc(n) = \frac{\sum_{i=i_s(n)}^{i_e(n)}(i-i_s(n)) \cdot x(i)}{\sum_{i=i_s(n)}^{i_e(n)} \cdot x(n)}

F0 (Pitch detection)
~~~~~~~~~~~~~~~~~~~~
An important feature of any periodic audio is it's pitch. Pitch is defined as
the perceived frequency of the signal. In order to determine the pitch of a
periodic signal, the fundamental frequency (F0) is estimated. There are many
methods developed for estimating the F0 of a signal. This program uses the
autocorrelation method. This method was chosen for it's simplicity and
reasonable versatility for a wide range of signals.

The f0 is calculated by first calculating the autocorellation of the signal
defined as:

.. math::
    R_i(m) = \sum_{n=1}^{W^L} x_i(n) x_i(n-m)

Then normalizing:

.. math::
    T_i(m) = \frac{R_i(m)}{\sqrt{\sum_{n=1}^{W^L}x_i(n)^2 \sum_{n=1}^{W^L}x_i(n-m)^2}}

The fundamental period of the signal is then calculated as the point between
:math:`T_{min}` and :math:`T_{max}` at which the correlated signal most closely matches the
original. :math:`T_{min}` and :math:`T_{max}` are defined as the minimum and maximum values of
the fundamental period.

.. math::
    x = arg\,max_{T_{min} \leq m \leq T_{max}} \{T_i(m)\}

In order to improve the accuracy of peak detection, parabolic interpolation is
used to estimate the peak's location with greater accuracy by using the peak
correlation and it's two closes neighbour's values to estimate the fractional
peak value.

The method for parabolic interpolation is defined as:

.. math::
    T_0^i = \frac{1}{2} \cdot \frac{\alpha - \gamma}{\alpha - 2\beta + \gamma} + x

    &\text{Where:} \\
    &\alpha = T_i(x-1) \\
    &\beta = T_i(x) \\
    &\gamma = T_i(x+1) \\
Ref: :cite:`quadinterp`

From this, the fundamental period the frequency is then calculated as:

.. math::
    f_0^i = \frac{1}{T_0^i}

Ref: :cite:`itaa2014`


FFT
~~~

Harmonic Ratio
~~~~~~~~~~~~~~
The harmonic ratio can be used to differentiate between noisy and periodic
signals. higher values suggest that the signal is more periodic (such as a sine
wave) and lower values represent less periodicity. This can be used as a form
of confidence measure in determining the validity of F0 values. it is
calculated as part of the F0 estimation algorithm as:

.. math::
    HR_i = max_{T_{min} \leq m \leq T_{max}}{\{T_i(m)\}}

Temporal Kurtosis
~~~~~~~~~~~~~~~~~

Peak Amplitude
~~~~~~~~~~~~~~

RMS
~~~

Spectral Centroid
~~~~~~~~~~~~~~~~~

Spectral Crest Factor
~~~~~~~~~~~~~~~~~~~~~

Spectral Flatness
~~~~~~~~~~~~~~~~~

Spectral Flux
~~~~~~~~~~~~~

Spectral Spread
~~~~~~~~~~~~~~~

Variance
~~~~~~~~

Zero-Crossing
~~~~~~~~~~~~~
.. bibliography:: refs.bib
