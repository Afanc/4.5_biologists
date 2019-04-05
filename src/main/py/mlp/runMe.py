#!/usr/bin/python
import numpy as np
import os

if __name__ == '__main__':
    reportFileName = 'MLP_test_report_out.txt'

    reportFile = open(reportFileName, 'w+')
    reportFile.write('******* Start MLP test ****** \n')
    reportFile.close()

    for batchSize in range(16, 70, 16):
        for learningRate in np.arange(1e-4, 1e-3, 3e-4):
            for dropout in np.arange(0.75, 1.05, 0.05):
                for hiddenWidth in range(64, 320, 64):
                    reportFile = open(reportFileName, 'a')
                    reportFile.write('\n\n**** Start MLP with: --batch_size '+str(batchSize)+' --learning_rate '+str(learningRate)+' --hidden_width '+str(hiddenWidth)+' --dropout '+str(dropout)+' **** \n')
                    reportFile.close()
                    cmd = 'python mlp_1.0.py --batch_size ' + str(batchSize) +' --learning_rate ' + str(learningRate) +' --hidden_width ' + str(hiddenWidth) +' --dropout ' + str(dropout) +' >> ' + reportFileName
                    os.system(cmd)
