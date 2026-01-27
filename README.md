# UavVisualLocalization

弱纹理环境下物理引导几何校验的无人机视觉定位完整实现。

## 注意

1. 本代码包含论文中的无人机两阶段定位，同时扩展了一个YOLO检测模块，与论文内容不冲突
2. 多线程并行化TopK加速代码将在论文被接受后提供

## 支持的模型

### 粗检索模型
- FSRA
- Sample4geo

### 细匹配模型
- SuperPoint+LightGlue
- SIFT+LightGlue 
- DISK+LightGlue
- ALIKED+LightGlue
- DoGHardNet+LightGlue
- LoFTR_fp16
- LoFTR_mp

## 论文状态

我们将于下周投稿《航空学报》。如使用本仓库代码，请在我们论文刊出后进行引用。
代码可能有错误，欢迎指正！
如有建议和疑问可以发邮件咨询：liutao23@njust.edu.cn

## 致谢

感谢以下仓库的作者提供的启发和灵感：

- https://github.com/Dmmm1997/FSRA.git
- https://github.com/Skyy93/Sample4Geo.git  
- https://github.com/magicleap/SuperPointPretrainedNetwork.git
- https://github.com/cvg/LightGlue.git
- https://github.com/zju3dv/LoFTR.git
- https://github.com/zju3dv/EfficientLoFTR.git
