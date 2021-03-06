# Chapter 3
## Gamma Correction
### Computer Color is Broken
[视频地址](https://www.youtube.com/watch?v=LKnqECcg6Gw)
#### 视频主要内容
* 2015年，多数绘图软件的多种颜色的交界处会显得很暗
* 人眼对亮度的感知是非线性的
* RGB都取最大值的50%（中灰），逻辑上是50%的白色，但物理上只有22%白色，取最大值的25%，只有5%的白色
<img src="https://img-blog.csdnimg.cn/20181206164249935.png" width="60%">
* 人类的视觉更擅长捕捉黑暗场景中的亮度变化
* 因此，为了节约储存，根据人眼感觉平均采样密度（将人眼感受到的中灰映射为0.5而非实际的0.2），曾经人们将原始像素的亮度开根号存储（通常这个值在1.8到2.2之间，称为gamma值），这样在感觉更敏感的低的亮度中采样了更多的点，在感觉相对不敏感的更亮的亮度中采样更少的点，可近似模拟人们对亮度均匀增加的感觉
* 在显示图像的时候，数值被还原后显示
<img src="https://img-blog.csdnimg.cn/2018120617074522.png" width="60%">
* 这带来了图像处理上的问题，之前的很多软件直接进行图像处理，并没有通过平方还原后再处理
* 假设从实际中捕捉了两个像素点，亮度分别为$x_1$，$x_2$，它们存储的值分别是$\sqrt{x_1}$，$\sqrt{x_2}$，为将它们叠加，本应该是先将存储值平方，后叠加为$\frac{x_1+x_2}{2}$，然后存储为$\sqrt\frac{x_1+x_2}{2}$，但疏忽的软件直接叠加并存储为$\sqrt\frac{\sqrt{x_1}+\sqrt{x_2}}{2}$，因此看起来就比预期要暗了许多，这是导致交界处显暗的原因

### 韦伯定律
人对自然界刺激的感知，是非线性的，按比例增加的刺激在人感觉来是线性的

举例就是，在黑暗的房间点燃一根蜡烛，刺激是明显的，但在有1000根蜡烛的房间点燃一根蜡烛，物理上的亮度增加是与之前相同的，但对人的刺激几乎微不足道

由此也可以发现，黑和白实际上并不是对称的，在黑中增加0.01亮度和在白中减少0.01亮度对人的感觉完全不同，之所以无法感觉到后者，是因为变化量在本身存在的量中微不足道，可知黑类似一种无的状态，白类似一种有的状态

人对音高（频率）、分贝等的感觉以及分级也是如此

### Gamma变换
人眼中的中灰其实是20%左右白而非50%白，这促使人们将20%对应的中灰存储为uint8（举例）中的128，防止原来情况下将中灰存储在20%*256这个位置时暗色存储数值范围小于亮色的情况。
**总的来说**
```mermaid
graph LR
A[原始亮度] -- 1/gamma --> B["存储值(RGB)"]
B --gamma--> C[显示器]
C --1/gamma--> D[人的感觉]
```
通常gamma取2.2，是实验得到的最终呈现近似线性的形状

[1] https://www.zhihu.com/question/27467127#answer-10413243 \
[2] https://blog.csdn.net/candycat1992/article/details/46228771 \
[3] https://blog.csdn.net/candycat1992/article/details/46228771
