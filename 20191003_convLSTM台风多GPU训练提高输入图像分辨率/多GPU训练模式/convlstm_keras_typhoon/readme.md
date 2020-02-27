"""typhoon predict trian and test by elesun"""
env:
	Linux 4.15.0-43-generic #46-Ubuntu x86_64 GNU/Linux
	Python 2.7.12 [GCC 5.4.0 20160609] on linux2
    scikit-image                       0.14.2     
    scikit-learn                       0.20.3 
    Pillow                             5.4.1 
    Keras                              2.2.4
    numpy                              1.16.1 
    tensorflow-gpu                     1.12.0 
    matplotlib                         2.2.3   
    pandas                             0.24.1  
    opencv-python                      4.0.0.21  
cmd:
	run
	sun@ubuntu:~/sun/radar_conv_lstm_501x501$ nohup python2 main_radar.py >out.log &

structure:
	data --source datasets like RAD_32528
	model -- saved trianed model
	output -- output constrast img
	main_radar.py -- main function control by mode test or train
	conv_lstm_network.py -- network disctption
	model_evaluate_predict.py -- load model and test
	preprocessing.py -- load dataset and process
