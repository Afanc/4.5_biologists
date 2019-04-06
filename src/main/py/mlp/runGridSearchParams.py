#!/usr/bin/python
import numpy as np
import os
import mlpMNIST as mlp

if __name__ == '__main__':
    nEpochs = 10
    reportFileName = 'MLP_test_report.csv'

    reportFile = open(reportFileName, 'w+')
    reportFile.write('***** MLP parameter test - MNIST data set - %d epochs per parameter combination ***** \n' % nEpochs)
    reportFile.write('final_accuracy_%, batch_size, learning_rate, hidden_width, dropout' + '\n')
    reportFile.close()
    i = 0
    print("Parameter optimization started \n   Creating report...")
    for batchSize in [64, 128, 192]:
        for learningRate in [1e-3, 3e-3]:
            for hiddenWidth in [512, 1024]:
                i += 1
                print("Parameter combination", i, "out of 12")
                #cmd = 'python mlp_1.0.py --batch_size ' + str(batchSize) +' --learning_rate ' + str(learningRate) +' --hidden_width ' + str(hiddenWidth) +' --dropout ' + str(dropout) +' >> ' + reportFileName
                #os.system(cmd)
                accuracy = mlp.trainAndTest(batch_size=batchSize,
                                 learning_rate=learningRate,
                                 hidden_width=hiddenWidth,
                                 n_epochs=nEpochs)
                reportFile = open(reportFileName, 'a')
                reportFile.write('%.2f, %d, %.4f, %d\n' % (accuracy, batchSize, learningRate, hiddenWidth))
                reportFile.close()