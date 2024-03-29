------------------------------------------------------------------------------------------------
About this algorithm:
An innovative method for automatic relocation of the focal depth of a local or regional earthquake, especially in the case that only a sparse network with stations located far away from the epicenter is available. For more details please refer to our paper below:
Yuan, J., Kao, H., & Yu, J. (2020). Depth-Scanning Algorithm: Accurate, Automatic, and Efficient Determination of Focal Depths for Local and Regional Earthquakes. Journal of Geophysical Research: Solid Earth, 125, e2020JB019430. https://doi.org/10.1029/2020JB019430


Framework:
  1. Automatic generation of synthetic waveforms for all possible depth phases.
  2. Match-filtering of all possible depth phases.
  3. Preliminary determination of the focal depth.
  4. Final solution based on travel time residuals.

Input:
  1. Three-component waveforms.
  2. Velocity model.

Output:
  Focal depth (median) 

------------------------------------------------------------------------------------------------

Installation requirements (recommend via Anaconda):
  1. Python 3.6.
  2. Basic packages: matplotlib, numpy, math, obspy, scipy, pandas.

------------------------------------------------------------------------------------------------

Data preparation for running DSA:
  1. Three-component waveforms (SAC) and corresponding response (XML).
        Waveform data can be one of the groups as follow:
            BH1/BH2/BHZ, BHE/BHN/BHZ, HH1/HH2/HHZ, HHE/HHN/HHZ, EHE/EHN/EHZ.

        SAC header at least should has corrected 'dist' and 'baz'. Please see examples in
            'Example1_PAPER-Section3.2',
            'Example2_2014-10-07-mw40-Oklahoma', etc.

        The response file can be downloaded by using the way like:
            https://docs.obspy.org/tutorial/code_snippets/stationxml_file_from_scratch.html

  2. Velocity model (TauP Toolkit format).
        Example can be seen in
            'ak135_Section3.2.nd' in 'Example1_PAPER-Section3.2'      

        More details see Section 5.1 in TauP manual at
            https://www.seis.sc.edu/downloads/TauP/taup.pdf
			
------------------------------------------------------------------------------------------------

Run test:
1. The default example is 'Example1_PAPER-Section3.2'. Run the command like:
  python DSA_1.0.py

2. Running other examples only needs to modify parameters in the 'DSA_SETTINGS.txt'.
    For example, to run the Oklahoma Mw=4.0 case you can copy all parameters from
          'DSA_SETTINGS_Example2.txt'
    to
          'DSA_SETTINGS.txt'

------------------------------------------------------------------------------------------------

See locating results:
1. After running, locating results are in the 'results' folder in the related waveform's directory.

2. A demo for plotting comparison of depth solutions is the python script named:
        'plot_comparisonFinalSolution_Oklahoma.py'

------------------------------------------------------------------------------------------------

If you use this software for your research, please cite:

Yuan, J., Kao, H., & Yu, J. (2020). Depth-Scanning Algorithm: Accurate, Automatic, and Efficient Determination of Focal Depths for Local and Regional Earthquakes. Journal of Geophysical Research: Solid Earth, 125, e2020JB019430. https://doi.org/10.1029/2020JB019430

------------------------------------------------------------------------------------------------
   
Any questions or advices? Please contact:
yuan_jianlong@126.com
honn.kao@canada.ca
j.yu@cdut.edu.cn

------------------------------------------------------------------------------------------------

Improvement at:

  2022.03.03
    Modifications due to new version of NumPy and Axes.hist:
      'normed' -> 'density' , 'np.int' -> 'int' , 'np.float' -> 'float'
      
  2022.08.10
    Suggestion from Ackerley Nicholas:
      Support loading: QuakeML and miniSEED. (see Example8_2014-10-07-16-51-13)
  
  2022.08.11
    Accelerated cross-correlation calculation by using 'numba' package.
