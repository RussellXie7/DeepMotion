# DeepMotion

DeepMotion is a novel solution the handwriting-to-text task. This system can recognize handwritten letters based on pen motion. We built it with an Arduino Uno R3, a MPU9250 motion sensor, and our own 3D printed molds. Using self-collected 5,000+ temporal motion data samples, we trained an RNN model with LSTM using TensorFlow and documented this study in following report. Ultimately, we hope to integrate this study with smart phone to let user visualized the recognized handwritten character on digital device and enhance the notetaking experience.

## Research Study Report

This research is documented in our [Research Paper](https://github.com/RussellXie7/DeepMotion/blob/master/paper/DeepMotion_Final_Report.pdf), where you can learn more about the methodology, experiments, and results of our study.


## Built With

* [Arduino IDE](https://www.arduino.cc/en/main/software) - The IDE used for hardware communication
* [TensorFlow](https://www.tensorflow.org/) - The framework used for Deep Learning Model design

## Authors

* **Wanze (Russell) Xie** - (https://github.com/russellxie7)
* **Chen Yang** 

## Acknowledgments

During this study, we have been fortunate to have the support and friendship of many individuals. First and foremost, we thank our professor, Zhuowen Tu. Over the weeks, Zhuowen has provided lots of great advises on our study, and he’s suggestions have great inspired us to explore the new application of state-of-the-art deep learning field. We would like to next say thank you to all the TAs of this class: Long Jin, Zeyu Chen, and Adilijiang Ainihaer . They have provided great support to our study and even helped us to create more data sample as our training and testing set.
