#!/usr/bin/python
import numpy as np
import os

if __name__ == '__main__':
    reportFileName = 'MLP_test_report_out.txt'

    reportFile = open(reportFileName, 'w+')
    reportFile.write('***** MLP parameter test ***** \n')
    reportFile.close()
    i=0
    print("Parameter optimization started \n   Creating report...")
    for batchSize in range(16, 70, 16):
        for learningRate in np.arange(1e-4, 1e-3, 3e-4):
            for dropout in np.arange(0.75, 1.05, 0.05):
                for hiddenWidth in range(64, 320, 64):
                    i+=1
                    print("iteration", i, "out of 336")
                    #reportFile = open(reportFileName, 'a')
                    #reportFile.write('batch_size\t'+str(batchSize)+'\tlearning_rate\t'+str(learningRate)+'\thidden_width\t'+str(hiddenWidth)+'\tdropout\t'+str(dropout)+'\n')
                    #reportFile.close()
                    cmd = 'python mlp_1.0.py --batch_size ' + str(batchSize) +' --learning_rate ' + str(learningRate) +' --hidden_width ' + str(hiddenWidth) +' --dropout ' + str(dropout) +' >> ' + reportFileName
                    reportFile = open(reportFileName, 'a')
                    reportFile.write("Final Accuracy : ")
                    reportFile.close()
                    os.system(cmd)
                    reportFile = open(reportFileName, 'a')
                    reportFile.write('%\tbatch_size: ' + str(batchSize) + '\tlearning_rate: ' + str(
                        learningRate) + '\thidden_width: ' + str(hiddenWidth) + '\tdropout: ' + str(dropout) + '\n')
                    reportFile.close()

                break
            break
        break
