![](Aspose.Words.2e190a72-26bc-4860-ab19-e602ca7f9b2b.001.png)

**Technical Assessment**

**Mashruk Jahangir**


Question 2:

I have used LSTM network on top of a CNN to implement a Video-based Action Recognition system. There are multiple other networks in use at the moment to achieve action recognition tasks. For example, RNN, Conv3D, etc. The reason I did not go for a RNN network because it is solely based on sequential data non-image data. Conv3D had proven to give remarkable results in many cases. However, it is computationally expensive. Hence, CNN+LTSM seemed like a wiser choice to go for. 

In this project, I broke down the UCF101 dataset into a smaller dataset with fewer classes in order to ease the process and training time constraint. I have uploaded the dataset for your convenience. I selected the following classes:

- Basketball
- Cricket
- Juggling Balls
- Billiard
- Archery
- Bowling
- High Jump

Basically, my model should give an insight regarding what sport is played in the video. The accuracy is amazing after having trained only 20-30 epochs due to my GPU and time constraints. 

Some snapshots of the results are given in the next page:   

`     `![](Aspose.Words.2e190a72-26bc-4860-ab19-e602ca7f9b2b.002.png)![](Aspose.Words.2e190a72-26bc-4860-ab19-e602ca7f9b2b.003.png)![](Aspose.Words.2e190a72-26bc-4860-ab19-e602ca7f9b2b.004.png)![](Aspose.Words.2e190a72-26bc-4860-ab19-e602ca7f9b2b.005.png)![](Aspose.Words.2e190a72-26bc-4860-ab19-e602ca7f9b2b.006.png)

Please run the inference.py script with the following arguments to try with your own video:

python inference.py --annotation\_path ../dataset/annotation/ucf101\_01.json  --dataset ucf101 --model cnnlstm --n\_classes 7 --resume\_path snapshots/latest.pth



