# Segmentation Pipeline on Meta Quest 3 with Unity

A real-time Augmented Reality computer vision pipeline for the **Meta Quest 3** built with
**Unity** and **Unity Sentis (Inference Engine)**. This project is part of the [Meta Quest Segmentation Pipeline Project](https://github.com/NguyenHoangNhat-git/Segmentation-Pipeline-on-Meta-Quest-3.git).

---

## How It Works
 
The pipeline runs as a loop on-device :

```
Passthrough Camera Frame
        │
        ▼
 YOLOv8n Detection (Sentis)  ──────────────────►  Bounding Boxes
        │                                              │
        │                                              ▼
        │                                  Raycast into world space
        │                                  (placed via EnvironmentRaycast)
        │                                              │
        ▼                                              ▼
  Object Info Panel  ◄────────────────────  Head-Gaze Selection
  (JSON lookup by class)                    (HeadGazeSelector)
                                                        │
                                                        ▼
                                          Crop ROI (+10% padding)
                                          Letterbox to model input
                                          (GazeCropExtractor)
                                                        │
                                                        ▼
                                     YOLO26n-sem Segmentation (Sentis)
                                          Dense per-pixel class map
                                                        │
                                                        ▼
                                          Mask Overlay in world space
                                          (SegmentationUiManager)
```


---

## Requirements
 
| Component | Version |
|---|---|
| Unity | 6000.0.38f1 or later (Unity 2022.3.58f1 LTS also supported by Meta's PCA samples) |
| Unity Sentis (Inference Engine) | 2.1+ |
| Meta XR SDK (Core / Interaction) | Latest |
| Mixed Reality Utility Kit (MRUK) | Latest — required for `PassthroughCameraAccess` |
| Newtonsoft Json for Unity | `com.unity.nuget.newtonsoft-json` |
| Headset | Meta Quest 3 (Passthrough Camera API is Quest 3 / 3S only) |
| Horizon OS | v74+ minimum; **v83+ required** for the 1280×1280 camera resolution |
 
---



## Project Structure
```bash
├── ...
├── ProcessingPipeline/
│   ├── EnvironmentRaycast/
│   │   └── Prefabs/
│   ├── HeadGaze/
│   │   ├── GazeCropExtractor.cs
│   │   └── HeadGazeSelector.cs
│   ├── ObjectInfo/
│   │   ├── object_info.cs
│   │   ├── ObjectInfoDatabase.cs
│   │   ├── ObjectInfoEntry.cs
│   │   └── ObjectInfoSelection.cs
│   ├── SentisInference/
│   │   ├── Editor/
│   │   │   ├── DetectionModelEditorConverter.cs
│   │   │   └── SegmentModelEditorConverter.cs
│   │   ├── Models/
│   │   │   ├── custom_detect_1280.onnx
│   │   │   └── custom_semantic_640.onnx
│   │   ├── Prefabs/
│   │   └── Scripts/
│   │   │   ├── MultiDetectionRunManager.cs
│   │   │   ├── MultiDetectionUiManager.cs
│   │   │   ├── SegmentationRunManager.cs
│   │   │   └── SegmentationUiManager.cs
│   └── Utility/
│       └── FPSCounter.cs
├── ...
├── .gitignore
└── README.md
```

---

## Models
The models recognize 16 classes from the 'Grasping in the Wild' dataset (https://universe.roboflow.com/iwrist/grasping-in-the-wild):
- Bowl
- CanOfCocaCola
- FryingPan
- Glass
- Jam
- Lid
- MilkBottle
- Mug
- OilBottle
- Plate
- Rice
- Saucepan
- Sponge
- Sugar
- VinegarBottle
- WashLiquid

---

## Usage / Controls
 
No hand controllers are required:
 
1. Put on the headset, go to Menu > Unknown Sources, open the app 'Segmentation Pipeline on MQ3', and grant camera permission when prompted on first launch.
2. Point your head at one of real-world objects that the detection model recognizes — a bounding box
   will appear anchored to it in 3D space.
3. Keep looking at a box (a reticle tracks your gaze direction) to select it — the box highlights
   and an info panel appears showing JSON-sourced details about that object class.
4. While the box stays selected, the region is periodically cropped and segmented; a colored
   mask overlay appears on top of the object once segmentation completes.
5. Look away to deselect — the info panel and mask overlay are hidden until a new object is
   selected.


---
## Credits
 
This project builds on Meta's official
[Unity-PassthroughCameraApiSamples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples)
(`MultiObjectDetection` sample).
 
Detection and segmentation models are built on [Ultralytics YOLO](https://docs.ultralytics.com/)
(YOLOv8n, YOLO26n).
 
JSON parsing uses [Newtonsoft.Json for Unity](https://docs.unity3d.com/Packages/com.unity.nuget.newtonsoft-json@latest).
 
---
