# 公开物体/3D资产数据集调研（截至 2026-03-05）

> 统计口径：优先官方项目页/官方文档/论文；若官方未明确披露，标注为“未明确”。

## 1. 关键信息总表

| 数据集 | 下载方式（公开可得） | 纹理情况 | 规模（物体/帧/抓取） |
|---|---|---|---|
| Objaverse 1.0 | 通过 `objaverse` Python API（底层托管于 Hugging Face）下载 UID 子集 | 多格式（GLTF/OBJ/FBX），通常含纹理但质量不一 | 官方 API 示例 `load_uids()` 为 **798,759** 个对象 |
| ShapeNetCore | ShapeNet 官网注册后下载；另有 HF 镜像入口 | 常见为 OBJ+MTL（部分类别含贴图） | **约 51,300** 模型，55 类 |
| ShapeNetSem | ShapeNet 官网注册后下载 | 以语义与物理属性标注为主，纹理覆盖未统一说明 | **12,000** 模型，270 类 |
| PartNet | 需先有 ShapeNet 账号；PartNet 官网+GitHub 脚本下载 | 以部件层级标注为主，纹理不是主目标 | **26,671** 模型，**573,585** part instances，24 类 |
| YCB Object & Model Set | YCB 官方 S3 站点直接下载（tgz） | 明确提供 texture-mapped meshes | 物理对象集 **77** 个；每对象 600 RGBD + 600 RGB |
| BigBIRD | 官网下载链接 | 提供 RGB 图像、RGB-D 点云、重建 mesh（纹理可由 RGB 支持） | **125** 物体（官网“currently 125”） |
| KIT ObjectModels | KIT ObjectModels Web UI 下载 | 提供 3D 数据+标定双目图像；纹理覆盖未统一说明 | 物体总量官方页面未给统一数字 |
| ContactDB | 官网“Code & Data / Raw Data”下载（含 IEEE DataPort） | 明确为带接触热图的 textured meshes | **50** 物体，**3750** meshes，**375K** RGB-D+thermal 帧 |
| OmniObject3D | 官网/官方 GitHub 提供下载指引 | 明确包含 textured meshes + point clouds + multiview/video | **6,000** 真实扫描物体，190 类 |
| Google Scanned Objects (GSO) / MuJoCo Scanned Objects (MSO) | 官方入口为 Gazebo Fuel collection（逐模型页面可下载）；社区常用镜像为 `mujoco_scanned_objects` GitHub 仓库 | 以真实扫描纹理网格为主（OBJ+MTL+贴图常见） | 官方 collection 页面未给统一“对象总数”与批量下载 API 说明 |
| HOPE（Image+Video） | 官方 GitHub `setup.py`（gdown）或 BOP 通道下载 | 明确提供 textured meshes | **28** 物体；HOPE-Video **10** 序列/2038 帧；HOPE-Image 188 test + 50 val |
| GraspNet-1Billion | 官网 datasets 页面（Google/Baidu/Jbox） | 含 object 3D models（models.zip）；纹理存在但非 benchmark 核心指标 | **88** 物体，190 场景，97,280 图像，>1.1B 抓取 |
| Dex-Net (v1.1/2.0) | Berkeley AUTOLAB data repository | Object Mesh Dataset 为 OBJ（纹理未强调） | v1.1 **1,500** 模型；Dex-Net 2.0 训练数据 **6.7M** 抓取样本 |
| ContactPose | 官网/官方 GitHub 下载 | 有对象接触图与对象 mesh；纹理相关数据可用 | **25** 物体，2306 抓取，>2.9M RGB-D 图像 |
| RealDex | 官方 GitHub 脚本下载（Google Drive） | 扫描重建 object meshes（纹理质量依扫描） | **52** 物体，约 **2.6K** 序列，约 **955K** 帧 |
| DexYCB | 官网提供整包或分卷 tar.gz 下载 | 使用 YCB 物体模型，具纹理 | **20** 个 YCB-Video 物体；**582K** RGB-D 帧；1000 序列 |
| ARCTIC | 官网 Code/Data/Competition 入口（需登录） | 重点是手-物交互时序与网格/接触，纹理非核心披露项 | **2.1M** 视频帧（双手+可动关节物体） |
| Visual Dexterity object assets | `dexenv` GitHub 指向 Hugging Face `assets.zip` 下载 | 为视觉重定向任务提供 object assets，纹理细节未统一披露 | 官方页面未给统一“对象数量”统计 |
| GRAB | 官网申请后下载 zip（含 `object_meshes.zip`） | 含对象 mesh 与接触标注；纹理不是论文主指标 | **51** 物体，10 被试，**1,622,459** 帧 |

## 2. 数据集之间物体重叠关系（可确认）

1. **PartNet 与 ShapeNet（明确重叠）**
- ShapeNet 官网新闻明确写到：PartNet 提供来自 ShapeNet 的细粒度部件标注（“PartNet provides ... from ShapeNet”）。

2. **DexYCB 与 YCB（明确重叠）**
- DexYCB 论文写明使用 **20 objects from the YCB-Video dataset**。

3. **GraspNet-1Billion 与 YCB（明确部分重叠）**
- GraspNet datasets 文档写明：object id `0-32` 与 `71` 来自 YCB。

4. **Dex-Net 与 KIT（明确重叠）**
- Dex-Net 文档写明 1500 模型中 **129** 来自 KIT Object Database。

## 3. 数据集之间重叠关系（高概率但公开证据不足）

- **GRAB / ARCTIC / ContactPose / ContactDB / HOPE** 之间都属于“日常物体抓取/交互”范畴，但官方页面通常未给出可机读的跨库对象 ID 映射，因此目前只能判断“语义类别有重叠（如 mug/scissors/laptop 等）”，难以确认“具体 mesh 级重合”。
- **Objaverse 与其他大规模 CAD 库** 在语义类别上高度重叠，但官方未提供与 ShapeNet/YCB/HOPE 的一对一对象映射表。

## 4. 备注（与你当前工程相关）

- 你当前工程目标（统一下载 -> 统一 raw -> 处理/水密化）中，最容易自动化接入的是：**YCB / Objaverse / DGN(BODex) / HOPE / DexYCB / GraspNet objects 子集**。
- 对于需要申请权限或登录的库（如 ShapeNet、PartNet、ARCTIC、GRAB），建议在工程里将下载模块拆成“自动下载 + 手动凭证下载占位器”。
- 对于 **GSO/MSO**：建议先把 Adapter 设计成“双后端”下载策略（`Fuel` 与 `GitHubMirror`）。`Fuel` 后端保留为官方优先路径（即使初期仅支持手动导入 URL 列表），`GitHubMirror` 后端用于可复现批量下载与 CI 验证。

## 5. 主要来源

- Objaverse 1.0 API: https://objaverse.allenai.org/docs/objaverse-1.0
- ShapeNet: https://shapenet.org/
- ShapeNet About/Download: https://shapenet.org/about
- PartNet: https://partnet.cs.stanford.edu/
- PartNet official repo: https://github.com/daerduoCarey/partnet_dataset
- YCB object set: https://www.ycbbenchmarks.com/object-set/
- YCB object models: https://www.ycbbenchmarks.com/object-models/
- YCB S3 index: https://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
- BigBIRD: https://rll.berkeley.edu/bigbird/
- KIT ObjectModels: https://archive.iar.kit.edu/Projects/ObjectModelsWebUI/index.php
- KIT DB intro: https://h2t.iar.kit.edu/1329.php
- ContactDB: https://contactdb.cc.gatech.edu/
- OmniObject3D: https://omniobject3d.github.io/
- OmniObject3D GitHub: https://github.com/omniobject3d/OmniObject3D
- Google Scanned Objects (Gazebo Fuel collection): https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research
- MuJoCo scanned objects mirror: https://github.com/kevinzakka/mujoco_scanned_objects
- HOPE repo: https://github.com/swtyree/hope-dataset
- GraspNet: https://graspnet.net/
- GraspNet datasets doc: https://graspnet.net/datasets.html
- Dex-Net data doc: https://berkeleyautomation.github.io/dex-net/data/data.html
- DexYCB: https://dex-ycb.github.io/
- DexYCB CVPR paper PDF: https://openaccess.thecvf.com/content/CVPR2021/papers/Chao_DexYCB_A_Benchmark_for_Capturing_Hand_Grasping_of_Objects_CVPR_2021_paper.pdf
- ContactPose: https://contactpose.cc.gatech.edu/
- RealDex repo: https://github.com/4DVLab/RealDex
- RealDex IJCAI paper PDF: https://www.ijcai.org/proceedings/2024/0758.pdf
- ARCTIC: https://arctic.is.tue.mpg.de/
- ARCTIC repo: https://github.com/zc-alexfan/arctic
- Visual Dexterity codebase: https://github.com/Improbable-AI/dexenv
- Visual Dexterity project page: https://taochenshh.github.io/projects/visual-dexterity
- GRAB project page: https://grab.is.tuebingen.mpg.de/
- GRAB repo: https://github.com/otaheri/GRAB
