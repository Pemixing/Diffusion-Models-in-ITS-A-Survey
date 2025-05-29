# Diffusion-Models-in-ITS-A-Survey
This repository investigates the applications of diffusion models in ITS, including autonomous driving, traffic simulation, traffic forecasting, and traffic safety. In particular, we review these applications based on criteria such as task, denoising condition, or model architecture. The classification is based on our survey [Diffusion Models for Intelligent Transportation Systems: A Survey](https://arxiv.org/pdf/2409.15816).

![image](https://github.com/user-attachments/assets/53ced87d-0abf-4825-b874-7770dbc0068e)


## 1. Diffusion Models for Autonomous Driving

## 1.1 Perception

| Paper | Task | Denoising Condition | Architecture | Datasets | Year | Open Source |
|-------|------|---------------------|--------------|----------|------|--------------|
| **DiffusionDet** [\[Chen et al. 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.pdf) | 2D object detection | conditioned on image feature | DDIM | CrowdHuman, COCO | 2023 ICCV | [Link](https://github.com/ShoufaChen/DiffusionDet) |
| **DetDiffusion** [\[Wang et al. 2024\]](https://arxiv.org/pdf/2403.13304) | 2D object detection | conditioned on perception-aware attributes | LDM | COCO | 2024 CVPR | —— |
| **DiffBEV** [\[Zou et al. 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28620) | BEV semantic segmentation, 3D object detection | conditioned on BEV feature | DDPM | nuScenes | 2024 AAAI | [Link](https://github.com/JiayuZou2020/DiffBEV) |
| **DDP** [\[Ji et al. 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Ji_DDP_Diffusion_Model_for_Dense_Visual_Prediction_ICCV_2023_paper.pdf) | BEV map segmentation, semantic segmentation, depth estimation | conditioned on image feature | DDIM | ADE20K, NYU-DepthV2, KITTI et al. | 2023 ICCV | [Link](https://github.com/JiYuanFeng/DDP) |
| **VPD** [\[Zhao et al. 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Unleashing_Text-to-Image_Diffusion_Models_for_Visual_Perception_ICCV_2023_paper.pdf) | semantic segmentation, image segmentation, depth estimation | conditioned on text | LDM | ADE20K, RefCOCO, NYU-DepthV2 | 2023 ICCV | [Link](https://github.com/wl-zhao/VPD) |
| [\[Chen et al. 2024\]](https://arxiv.org/pdf/2403.04700) | multi-object tracking | conditioned on text | LDM | MOT20 et al. | 2024 CVPR | [Link](https://github.com/chen-si-jia/Trajectory-Long-tail-Distribution-for-MOT) |
| [\[Luo et al. 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28192) | multi-object tracking | conditioned on two adjacent raw images | DDPM | MOT20 et al. | 2024 AAAI | [Link](https://github.com/RainBowLuoCS/DiffusionTrack) |
| [\[Xie et al. 2024\]](https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_DiffusionTrack_Point_Set_Diffusion_Model_for_Visual_Object_Tracking_CVPR_2024_paper.pdf) | object tracking | unconditional | DDIM | GOT-10k, LaSOT | 2024 CVPR | [Link](https://github.com/VISION-SJTU/DiffusionTrack) |
| [\[Luo et al. 2021\]](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Diffusion_Probabilistic_Models_for_3D_Point_Cloud_Generation_CVPR_2021_paper.pdf) | 3D point cloud generation | conditioned on shape latent | DDPM | ShapeNet | 2021 CVPR | [Link](https://github.com/luost26/diffusion-point-cloud) |
| **DiffuMask** [\[Wu et al. 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_DiffuMask_Synthesizing_Images_with_Pixel-level_Annotations_for_Semantic_Segmentation_Using_ICCV_2023_paper.pdf) | semantic segmentation, perception data augmentation | conditioned on text | LDM | VOC, ADE20K, Cityscapes | 2023 ICCV | [Link](https://github.com/weijiawu/DiffuMask) |
| **DatasetDM** [\[Wu et al. 2023\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/ab6e7ad2354f350b451b5a8e14d04f51-Paper-Conference.pdf) | perception data augmentation | conditioned on text | LDM | COCO et al. | 2023 NeurIPS | [Link](https://github.com/showlab/DatasetDM) |

---

## 1.2 Trajectory Prediction

| Paper | Task | Denoising Condition | Architecture | Datasets | Year | Open Source |
|-------|------|---------------------|--------------|----------|------|--------------|
| **MID** [\[Gu et al. 2022\]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Stochastic_Trajectory_Prediction_via_Motion_Indeterminacy_Diffusion_CVPR_2022_paper.pdf) | human trajectory prediction | conditioned on observed trajectories | DDPM | SDD, ETH, UCY | 2022 CVPR | [Link](https://github.com/gutianpei/MID) |
| **LED** [\[Mao et al. 2023\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper.pdf) | human trajectory prediction, speed up | conditioned on observed trajectories | LED | SDD et al. | 2023 CVPR | [Link](https://github.com/MediaBrain-SJTU/LED) |
| **SingularTrajectory** [\[Bae et al. 2024\]](https://openaccess.thecvf.com/content/CVPR2024/papers/Bae_SingularTrajectory_Universal_Trajectory_Predictor_Using_Diffusion_Model_CVPR_2024_paper.pdf) | human trajectory prediction, speed up | conditioned on observed scene | DDIM | ETH et al. | 2024 CVPR | [Link](https://github.com/inhwanbae/SingularTrajectory) |
| **IDM** [\[Liu et al. 2024\]](https://arxiv.org/pdf/2403.09190) | human trajectory prediction, speed up | conditioned on observed trajectories, endpoint | DDPM | SDD et al. | 2024 arXiv | —— |
| **LADM** [\[Lv et al. 2024\]](https://ieeexplore.ieee.org/document/10466609) | human trajectory prediction, speed up | conditioned on coarse future trajectory | VAE + DDPM | ETH et al. | 2024 TIM | —— |
| **BCDiff** [\[Li et al. 2024\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/2e57e2c14232a7b99cf76213e190822d-Paper-Conference.pdf) | human trajectory prediction, instantaneous prediction | conditioned on gate | DDPM | SDD et al. | 2024 NeurIPS | —— |
| **MotionDiffuser** [\[Jiang et al. 2023\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_MotionDiffuser_Controllable_Multi-Agent_Motion_Prediction_Using_Diffusion_CVPR_2023_paper.pdf) | multi-agent prediction | conditioned on observed scene, constraints; classifier guidance | LDM | WOMD | 2023 CVPR | —— |
| **SceneDiffusion** [\[Balasubramanian et al. 2023\]](https://ieeexplore.ieee.org/abstract/document/10422482) | multi-agent prediction | conditioned on observed scene, interval time; unconditional | LDM | Argoverse | 2023 ITSC | —— |
| **Equidiff** [\[Chen et al. 2023\]](https://ieeexplore.ieee.org/document/10421892) | vehicle trajectory prediction | conditioned on observed trajectories, interactions | DDPM | NGSIM | 2023 ITSC | —— |
| [\[Yao et al. 2023\]](https://ieeexplore.ieee.org/document/10363970) | vehicle trajectory prediction | conditioned on observed trajectories, map | DDPM | Argoverse2 | 2023 CSIS-IAC | —— |

---

## 1.3 Planning and Decision Making

| Paper | Task | Denoising Condition | Architecture | Datasets | Year | Open Source |
|-------|------|---------------------|--------------|----------|------|--------------|
| **Diffuser** [\[Janner et al. 2022\]](https://github.com/jannerm/diffuser) | behavior planning | unconditional, classifier guidance | ADM | D4RL | 2022 ICML | [Link](https://github.com/jannerm/diffuser) |
| **Decision Diffuser** [\[Ajay et al. 2023\]](https://arxiv.org/abs/2301.00007) | decision making, behavior planning | conditioned on rewards, constraints, skills; classifier-free guidance | ADM | D4RL | 2023 ICLR | —— |
| **MPD** [\[Carvalho et al. 2023\]](https://github.com/jacarvalho/mpd-public) | motion planning | unconditional, classifier guidance | DDPM | PointMass2D | 2023 IROS | [Link](https://github.com/jacarvalho/mpd-public) |
| **Diffusion-ES** [\[Yang et al. 2024\]](https://github.com/bhyang/diffusion-es) | motion planning | unconditional | truncated DDPM | nuPlan | 2024 CVPR | [Link](https://github.com/bhyang/diffusion-es) |
| **Drive-WM** [\[Wang et al. 2024\]](https://github.com/BraveGroup/Drive-WM) | motion planning, multiview video generation | conditioned on adjacent views | VLDM | nuScenes | 2024 CVPR | [Link](https://github.com/BraveGroup/Drive-WM) |
| **GenAD** [\[Yang et al. 2024\]](https://github.com/OpenDriveLab/DriveAGI) | motion planning, multiview video generation | conditioned on past frame, text | VLDM | WOMD et al. | 2024 CVPR | [Link](https://github.com/OpenDriveLab/DriveAGI) |

---

## 2. Diffusion Models for Traffic Simulation

### 2.1 Trajectory Generation

| Paper | Task | Denoising Condition | Architecture | Datasets | Year | Open Source |
|-------|------|----------------------|--------------|----------|------|--------------|
| **CTG** [\[Zhong et al. 2023\]](https://github.com/NVlabs/CTG) | vehicle trajectory generation | conditioned on observed scene; <br> STL-based guidance | ADM | nuScenes | 2023 ICRA | [CTG](https://github.com/NVlabs/CTG) |
| **CTG++** [\[Zhong et al. 2023\]](https://github.com/NVlabs/CTG) | multi-agent trajectory generation | conditioned on observed scene; <br> language-based guidance | ADM | nuScenes | 2023 CoRL | [CTG++](https://github.com/NVlabs/CTG) |
| **Dragtraffic** [\[Wang et al. 2024\]](https://github.com/chantsss/Dragtraffic) | multi-agent trajectory generation | conditioned on initial scene, text | LED | WOMD | 2024 IROS | [Dragtraffic](https://github.com/chantsss/Dragtraffic) |
| **DJINN** [\[Niedoba et al. 2024\]]() | multi-agent trajectory generation | conditioned on arbitrary state; <br> classifier-free guidance; <br> behavior classes guidance | EDM | Argoverse <br> INTERACTION | 2024 NeurIPS | — |
| **Pronovost et al. 2023** | multi-agent trajectory generation | conditioned on map, tokens | EDM <br> LDM | Argoverse2 | 2023 NeurIPS | — |
| **Rempe et al. 2023** [\[trace\]](https://github.com/nv-tlabs/trace) [\[pacer\]](https://github.com/nv-tlabs/pacer) | human trajectory generation | conditioned on observed scene; <br> classifier-free guidance | ADM | ETH, nuScenes | 2023 CVPR | [trace](https://github.com/nv-tlabs/trace), [pacer](https://github.com/nv-tlabs/pacer) |

---

### 2.2 Traffic Scenario Generation

| Paper | Task | Denoising Condition | Architecture | Datasets | Year | Open Source |
|-------|------|----------------------|--------------|----------|------|--------------|
| **FDM** [\[Harvey et al. 2022\]]() | image-based driving scenario generation | conditioned on previously sampled frames | FDM | Carla | 2022 NeurIPS | — |
| **GAIA-1** [\[Hu et al. 2023\]]() | image-based driving scenario generation | conditioned on past image, text, action tokens; <br> classifier-free guidance | VDM <br> FDM | real-world dataset | 2023 arXiv | — |
| **DriveDreamer** [\[Wang et al. 2023\]](https://github.com/JeffWang987/DriveDreamer) | image-based driving scenario generation | conditioned on image, road structure, text | LDM <br> VLDM | nuScenes | 2023 arXiv | [DriveDreamer](https://github.com/JeffWang987/DriveDreamer) |
| **DriveDreamer-2** [\[Zhao et al. 2024\]](https://github.com/f1yfisher/DriveDreamer2) | image-based driving scenario generation | conditioned on structured info by LLMs, text | EDM | nuScenes | 2024 arXiv | [DriveDreamer2](https://github.com/f1yfisher/DriveDreamer2) |
| **Panacea** [\[Wen et al. 2024\]](https://github.com/wenyuqing/panacea) | image-based driving scenario generation | conditioned on image, text, BEV sequence | LDM <br> DDIM | nuScenes | 2024 CVPR | [panacea](https://github.com/wenyuqing/panacea) |
| **DrivingDiffusion** [\[Li et al. 2023\]](https://github.com/shalfun/DrivingDiffusion) | image-based driving scenario generation | conditioned on key-frame, optical flow prior, text, 3D layout | VDM <br> LDM | nuScenes | 2023 arXiv | [DrivingDiffusion](https://github.com/shalfun/DrivingDiffusion) |
| **WoVoGen** [\[Lu et al. 2023\]](https://github.com/fudan-zvg/WoVoGen) | image-based driving scenario generation | conditioned on past world volumes, actions, text, 2D image feature | LDM | nuScenes | 2023 arXiv | [WoVoGen](https://github.com/fudan-zvg/WoVoGen) |
| **LiDMs** [\[Ran et al. 2024\]](https://github.com/hancyran/LiDAR-Diffusion) | point cloud-based driving scenario generation | unconditional; <br> conditioned on arbitrary data | LDM | nuScenes <br> KITTI-360 | 2024 CVPR | [LiDAR-Diffusion](https://github.com/hancyran/LiDAR-Diffusion) |
| **Copilot4D** [\[Zhang et al. 2023\]]() | point cloud-based driving scenario generation | conditioned on past observations, actions; <br> classifier-free guidance | D3PM <br> ADM | nuScenes et al. | 2024 ICLR | — |

---

### 2.3 Traffic Flow Generation

| Paper | Task | Denoising Condition | Architecture | Datasets | Year | Open Source |
|-------|------|----------------------|--------------|----------|------|--------------|
| **KSTDiff** [\[Zhou et al. 2023\]](https://github.com/tsinghua-fib-lab/KSTDiff-Urban-flow-generation) | traffic flow generation | conditioned on urban knowledge graph, region feature, volume estimator | CARD | real-world dataset | 2023 SIGSPATIAL | [KSTDiff](https://github.com/tsinghua-fib-lab/KSTDiff-Urban-flow-generation) |
| **DiffTraj** [\[Zhu et al. 2023\]](https://github.com/Yasoz/DiffTraj) | GPS trajectory generation | conditioned on trip region, departure time; <br> classifier-free guidance | DDIM, ADM | real-world dataset | 2023 NeurIPS | [DiffTraj](https://github.com/Yasoz/DiffTraj) |
| **Diff-RNTraj** [\[Wei et al. 2024\]]() | GPS trajectory generation | conditioned on road network | DDPM | real-world dataset | 2024 arXiv | — |
| **ChatTraffic** [\[Zhang et al. 2023\]](https://github.com/ChyaZhang/ChatTraffic) | traffic flow generation | conditioned on text | LDM | text-traffic pairs dataset | 2024 arXiv | [ChatTraffic](https://github.com/ChyaZhang/ChatTraffic) |
| **Rong et al. 2023** | origin-destination flow generation | conditioned on node feature, edge feature | DDPM, ADM | real-world dataset | 2023 arXiv | — |

---

## 3. Diffusion Models for Traffic Forecasting

### 3.1 Traffic Flow Forecasting

| Paper | Task | Condition | Architecture | Dataset | Year | Open Source |
|-------|------|-----------|--------------|---------|------|-------------|
| **DiffSTG** [\[Wen et al. 2023\]](https://github.com/wenhaomin/DiffSTG) | traffic flow forecasting | conditioned on past graph signals, graph structure | NCSN | PEMS et al. | 2023 GIS | [DiffSTG](https://github.com/wenhaomin/DiffSTG) |
| **SpecSTG** [\[Lin et al. 2024\]](https://anonymous.4open.science/r/SpecSTG/README.md) | traffic flow forecasting <br> traffic speed forecasting | conditioned on past graph signals feature, adjacency matrix | DDPM | PEMS et al. | 2024 arXiv | [SpecSTG](https://anonymous.4open.science/r/SpecSTG/README.md) |
| **DiffUFlow** [\[Zheng et al. 2023\]]() | traffic flow forecasting | conditioned on pass feature map, coarse-grained flow map, semantic features | DDPM | real-world dataset | 2023 CIKM | — |
| **Xu et al. 2023** | traffic flow forecasting | unconditional | DDPM | real-world dataset | 2023 ICASSP | — |
| **ST-SSPD** [\[Lablack et al. 2023\]]() | traffic flow forecasting | conditioned on past data points, temporal encoding, node identifier | DDPM | METR-LA et al. | 2023 MobiArch | — |
| **Difforecast** [\[Chi et al. 2023\]]() | traffic flow forecasting <br> image generation | conditioned on past S-T image | DDPM | real-world dataset | 2023 BigData | — |

---

### 3.2 Travel Time Estimation

| Paper | Task | Condition | Architecture | Dataset | Year | Open Source |
|-------|------|-----------|--------------|---------|------|-------------|
| **Lin et al. 2023** | origin-destination travel time estimation | conditioned on origin, destination, departure time | DDPM | real-world dataset | 2023 MOD | — |

---

## 4. Diffusion Models for Traffic Safety

### 4.1 Traffic Anomaly Detection

| Paper | Task | Condition | Architecture | Dataset | Year | Open Source |
|-------|------|-----------|--------------|---------|------|-------------|
| **DiffTAD** [\[Li et al. 2024\]]() | trajectory anomaly detection | unconditional | DDIM | NGSIM | 2024 KBS | — |
| **VAD** [\[Yan et al. 2023\]]() | video anomaly detection | unconditional; <br> conditioned on original features | LDM, DDIM | CUHK Avenue et al. | 2023 ICCV | — |

---

### 4.2 Traffic Accident Prevention

| Paper | Task | Condition | Architecture | Dataset | Year | Open Source |
|-------|------|-----------|--------------|---------|------|-------------|
| **AdVersa-SD** [\[Fang et al. 2024\]](https://github.com/jeffreychou777/LOTVS-MM-AU) | accident video understanding <br> accident preventing | conditioned on text, bounding boxes | LDM | MM-AU | 2024 CVPR | [MM-AU](https://github.com/jeffreychou777/LOTVS-MM-AU) |

