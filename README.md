# jaylyrics_generation_tensorflow

此项目fork自https://github.com/zzw922cn/jaylyrics_generation_tensorflow ，感谢zzw922cn的奉献！
使用基于LSTM的swq2seq模型并结合注意力机制来生成周杰伦歌词，加入了十多部中文小说以及散文歌词等进行了语言模型的预训练，然后再使用周杰伦的歌词来训练模型进而实现抽样再生。详情请关注我的微信公众号：deeplearningdigest
在此基础之上进行了修改，增加了对Windows的支持，Python3.5下运行成功。
## 训练
`python train.py `

## 抽样生成单词
`python sample.py `

使用`从前`作为种子序列，生成歌词如下：
```
从前进开封 出水芙蓉加了星
在狂风叫我的爱情 让你更暖
心上人在脑海别让人的感觉
想哭 是加了糖果
船舱小屋上的锈却现有没有在这感觉
你的美 没有在
我却很专心分化
还记得
原来我怕你不在 我不是别人
要怎么证明我没力气 我不用 三个马蹄
生命潦草我在蓝脸喜欢笑
中古世界的青春的我
而她被芒草割伤
我等月光落在黑蝴蝶
外婆她却很想多笑对你不懂 我走也绝不会比较多
反方向左右都逢源不恐 只能永远读着对白
```

由于tf在Windows上只支持py35,所以如果已经安装了py27但又不想重新安装的朋友，可以通过http://blog.csdn.net/infin1te/article/details/50445217 方法，安装py35,然后从http://www.lfd.uci.edu/~gohlke/pythonlibs/ 下载tf的whl文件安装。
此项目，目前没有使用pre-trained,大家可以自己更改就行了！


我的这个是用了中医中的方剂进行训练的，后续再加入新的思想吧！
