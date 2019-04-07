#!/usr/bin/python
import numpy as np
import os
import mlpMNIST as mlp

if __name__ == '__main__':
    maxEpochs = 15
    delta_accuracy = 1.
    reportFileName = 'MLP_test_report.csv'

    reportFile = open(reportFileName, 'w+')
    reportFile.write('***** MLP parameter test - MNIST data set - max %d epochs  per parameter combination  and delta_accuracy of %.2f %% ***** \n'
                     % (maxEpochs, delta_accuracy))
    reportFile.write('final_accuracy_%, batch_size, learning_rate, hidden_width, epochs \n')
    reportFile.close()
    i = 0
    print("Parameter optimization started \n   Creating report...")
    bs_range = [64, 128, 192, 256, 512, 1024]
    for batchSize in bs_range:
        lr_range = [1e-3, 3e-3, 5e-3, 7e-3, 9e-3]
        for learningRate in lr_range:
            hw_range = [256, 512, 1024, 2048, 4096]
            for hiddenWidth in hw_range:
                i += 1
                print("Parameter combination", i, "out of ", len(bs_range) * len(lr_range) * len(hw_range))
                #cmd = 'python mlp_1.0.py --batch_size ' + str(batchSize) +' --learning_rate ' + str(learningRate) +' --hidden_width ' + str(hiddenWidth) +' --dropout ' + str(dropout) +' >> ' + reportFileName
                #os.system(cmd)
                (accuracy, epochs) = mlp.trainAndTest(batch_size=batchSize,
                                 learning_rate=learningRate,
                                 hidden_width=hiddenWidth,
                                 n_epochs=maxEpochs,
                                 delta_accuracy=delta_accuracy)
                reportFile = open(reportFileName, 'a')
                reportFile.write('%.2f, %d, %.4f, %d, %d\n' % (accuracy, batchSize, learningRate, hiddenWidth, epochs))
                reportFile.close()