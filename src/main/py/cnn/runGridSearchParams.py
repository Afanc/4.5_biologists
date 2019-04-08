#!/usr/bin/python
import numpy as np
import os
import model_task2c as cnn

if __name__ == '__main__':
    maxEpochs = 1
    reportFileName = 'CNN_test_report.csv'

    reportFile = open(reportFileName, 'w+')
    reportFile.write('***** CNN parameter test - MNIST data set - max %d epochs  per parameter combination ***** \n'
                     % (maxEpochs))
    reportFile.write('final_accuracy_%, batch_size, learning_rate, gamma\n')
    reportFile.close()
    i = 0
    print("Parameter optimization started \n   Creating report...")
    bs_range = [16,32,64]
    for batchSize in bs_range:
        lr_range = [1e-3, 3e-3, 5e-3]
        for learningRate in lr_range:
            gamma_range = [0.1, 0.3, 0.5]
            for gamma in gamma_range:
                i += 1
                print("Parameter combination", i, "out of ", len(bs_range) * len(lr_range) * len(gamma_range))
                accuracy = cnn.trainAndTest(batch_size=batchSize,
                                 learning_rate=learningRate,
                                 n_epochs=maxEpochs,
                                 g=gamma)
                reportFile = open(reportFileName, 'a')
                reportFile.write('%.3f, %d, %.4f, %.1f\n' % (accuracy, batchSize, learningRate, gamma))
                reportFile.close()
