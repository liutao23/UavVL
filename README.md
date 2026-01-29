# UavVisualLocalization

弱纹理环境下物理引导几何校验的无人机视觉定位完整实现。

## 注意

1. 本代码包含论文中的无人机两阶段定位，同时扩展了一个YOLO检测模块，与论文内容不冲突
2. 多线程并行化TopK加速代码将在论文被接受后提供
3. 原始数据太大我在仓库里面删除了，一个示例文件大概5g，有需要的发邮件向我索取，然后将压缩包解压在工作目录就好！也可以等稿件录用以后，我们公布全部的卫星图处理脚本，自行处理也是可以的
4. 有任何问题都可以发邮件询问，看到都会及时回复，也可以在issue里面提问
 <img width="131" height="153" alt="image" src="https://github.com/user-attachments/assets/e6400dba-cab3-492d-a89c-340081f8e606" />

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

我们论文已经投稿。如使用本仓库代码，请在我们论文刊出后进行引用。
代码可能有错误，欢迎指正！
如有建议和疑问可以在issue咨询：

## 致谢

感谢以下仓库的作者提供的启发和灵感：

- https://github.com/Dmmm1997/FSRA.git
- https://github.com/Skyy93/Sample4Geo.git  
- https://github.com/magicleap/SuperPointPretrainedNetwork.git
- https://github.com/cvg/LightGlue.git
- https://github.com/zju3dv/LoFTR.git
- https://github.com/zju3dv/EfficientLoFTR.git
- https://github.com/danini/magsac.git
## 引用（待更新）

