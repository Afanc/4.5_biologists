#!/usr/bin/python
import numpy as np
import os

if __name__ == '__main__':
    reportFileName = 'MLP_test_report_out.txt'

    reportFile = open(reportFileName, 'w+')
    reportFile.write('***** MLP parameter test ***** \n')
    reportFile.write('final_accuracy, batch_size, learning_rate, hidden_width, dropout' + '\n')
    reportFile.close()
    i=0
    print("Parameter optimization started \n   Creating report...")
    for batchSize in [64, 128, 192]:
        for learningRate in [1e-3, 3e-3]:
            for dropout in [1, 0.9, 0.8]:
                for hiddenWidth in [512, 1024]:
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