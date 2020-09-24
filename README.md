# Covid_detection_CNN

AIRO project: Vision & Perception course

<a href="https://www.dis.uniroma1.it/"><img src="http://www.dis.uniroma1.it/sites/default/files/marchio%20logo%20eng%20jpg.jpg" width="500"></a>

## Approach 
Covid-non Covid detection, with two experiments: RES_NET transfer learning vs custom CNN trained from scratch

![](cnn.gif)

## Team
* Flavio Lorenzi <a href="https://github.com/FlavioLorenzi"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/1024px-Octicons-mark-github.svg.png" width="30"></a>
<a href="https://www.linkedin.com/in/flavio-lorenzi-875982171/"><img src="https://www.tecnomagazine.it/tech/wp-content/uploads/2013/05/linkedin-aggiungere-immagini.png" width="30"></a>

* Nicolò Mantovani <a href="https://github.com/Nicodman"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/1024px-Octicons-mark-github.svg.png" width="30"></a>

* Michele Ciciolla <a href="https://github.com/micheleciciolla"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/1024px-Octicons-mark-github.svg.png" width="30"></a>



## Documentation
Idea, part of dataset and of the transfer learning experiment is taken from here: (shervinmin/DeepCovid) <a href="https://github.com/shervinmin/DeepCovid"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/1024px-Octicons-mark-github.svg.png" width="30"></a>

Dataset, still too small, it has been increased thanks to some Data Augmentation techniques described in the document (pdf).


COVID
![SC2 Image](covid.jpg)

NO COVID
![SC2 Image](non-covid.jpg)

## Results 
Very good results for each experiment, with an inference accuracy bigger than 0.9 : this is the consequence of a careful tuning of the parameters, but above all, this is due to the fact that it is a Binary Classification (easier to train than a multiclass one).

Custom CNN training-test plot:


![SC2 Image](result1.png)

Confusion matrix of the Transfer Learning inference experiment, given a blind balanced dataset of 100 imgs:


![SC2 Image](result2.png)
