# mEtRoBoOm
A convolutional neural network used to predict music genre. Created alongside partner [Armaan Lalani](https://github.com/armaanlalani)



## Table of Contents
* [Background](#back)
* [Technologies Used](#tech)
 

<a name="back"></a>
### Background 

The overarching goal of this project is to develop a neural network capable of classifying the genre of an inputted mp3 file. The inputted mp3 file is first converted to a mel-spectrogram which will then be passed through a neural network to predict the genre. The scope of the genres is condensed to one of 7 genres:
  * Hip Hop
  * Pop
  * RnB
  * Rock
  * Latin
  * EDM
  * Country
  
  Mel-spectrograms are extremely important when conducting auditory analysis as it essentially extracts all of the most important features of an audio clip into a singular image, where each pixel represents something specific about the audio. This process is accomplished through a series of transforms including:
  * A Fourier transform of the signal over numerous equally-sized windows
  * Split the entire frequency spectrum into many evenly-distributed frequencies, where the frequencies ‘sound’ equally distanced to one another based on human hearing.
  * At this point one has a spectrogram; thus, to obtain a mel-spectrogram, one transforms on frequency spectrum into a mel spectrum. 

  Before finalizing this topic, one very important aspect of the background we had to consider is that music genre classification is extremely subjective; there is ambiguity in deciding exactly what genre some songs may belong to. Music genre subjectivity is extremely apparent in the genres of hip-hop, r&b, and pop which is something we expected to see in the results of the project.


<a name="tech"></a>
### Technologies Used
  * Python 3.x
  * PyTorch 1.6.0
  * Librosa 

    
