# fk-beamforming-merapi

This was my first Python project when I was working on my college final project. Yes, it's messy, and I'm still learning. Using frequency wavenumber beamforming (fk-beamforming) calculations based on seismic data from four stations, I calculated the back azimuth of rockfall and pyroclastic flow in Merapi Mountain, Yogyakarta and compared with the visual observations then interpreted to determine the source and direction of the pyroclastic flow. Cheers!

## Processing

### Seismic data filtering

On this project, the seismic signal was trim in range between the event that occurred using obspy from an Excel (input it with Pandas) file in this folder.
The seismic signal was filtered using a bandpass filter from 1.0 Hz to 12.0 Hz (typical rockfall signal from Merapi).

![Semantic description of image](/assets/img/Data-Trimming.png "Data Trimming (a) before (b) after")<br>
_Data Trimming (a) before (b) after_
<br>
![Semantic description of image](/assets/img/Data_Filtering.png "Data Filtering 1.0 Hz to 12.0Hz (a) before (b) after") <br>
_Data Trimming (a) before (b) after_

### FK Beamforming

The fk beamforming analysis required several parameters because, in this study, the back azimuth and slowness values were unknown, so the grid search method was used.
Which is then carried out by trial, estimating the energy of the beamform signal at each of these slowness values in each time window. You can read more detailed calculations in the Obspy documentation and paper from Rost, S & Thomas, C (2002).

![Semantic description of image](/assets/img/FK-Beamforming.png "FK-Beamforming") <br>
_Back Azimuth as result from FK-Beamforming_

### Particle Motion

The starting point of the recording data is plotted in yellow on the particle motion graph and the end of the recording data (0.5s from the starting point) is plotted in purple along with the color gradation.

![Semantic description of image](/assets/img/Particle-Motion.png "Particla Motion") <br>
_Visualization of particle motion from each station showing direction of each data record_

### Visualization

The FK-beamforming analysis results will be divided based on the direction of occurrence of a pyroclastic flow from the summit. Particle motion analyses will also support the FK-Beamforming results. The polar diagram represents the distribution of energy in color levels, while the gray numbers in the polar diagram describe the slowness value for each energy.

note: I attached my final project to this repository, but sorry, it's only in Indonesian.
